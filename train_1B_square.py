#!/usr/bin/env python3
import argparse
import gc
import logging
import math
import os
import pickle
import random
import shutil
import sys

import accelerate
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms.functional as TF
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import DDIMScheduler, FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (EMAModel,
                                      compute_density_for_timestep_sampling,
                                      compute_loss_weighting_for_sd3)
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from torch import nn
from torch.utils.data import RandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import datasets
import librosa
from pathlib import Path
import imageio
import torchvision
from transformers import Wav2Vec2Model, Wav2Vec2Processor

from wan.dataset.talking_video_dataset_fantasy import LargeScaleTalkingFantasyVideos

from wan.models.wan_fantasy_transformer3d_1B import WanTransformer3DFantasyModel
from wan.models.wan_text_encoder import WanT5EncoderModel
from wan.models.wan_vae import AutoencoderKLWan
from wan.models.wan_image_encoder import CLIPModel
from wan.pipeline.wan_inference_pipeline_fantasy import WanI2VFantasyPipeline

from wan.utils.discrete_sampler import DiscreteSampling
from wan.utils.utils import get_image_to_video_latent, save_videos_grid


def filter_kwargs(cls, kwargs):
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return filtered_kwargs


def get_random_downsample_ratio(sample_size, image_ratio=[],
                                all_choices=False, rng=None):
    def _create_special_list(length):
        if length == 1:
            return [1.0]
        if length >= 2:
            first_element = 0.75
            remaining_sum = 1.0 - first_element
            other_elements_value = remaining_sum / (length - 1)
            special_list = [first_element] + [other_elements_value] * (length - 1)
            return special_list

    if sample_size >= 1536:
        number_list = [1, 1.25, 1.5, 2, 2.5, 3] + image_ratio
    elif sample_size >= 1024:
        number_list = [1, 1.25, 1.5, 2] + image_ratio
    elif sample_size >= 768:
        number_list = [1, 1.25, 1.5] + image_ratio
    elif sample_size >= 512:
        number_list = [1] + image_ratio
    else:
        number_list = [1]

    if all_choices:
        return number_list

    number_list_prob = np.array(_create_special_list(len(number_list)))
    if rng is None:
        return np.random.choice(number_list, p=number_list_prob)
    else:
        return rng.choice(number_list, p=number_list_prob)


def resize_mask(mask, latent, process_first_frame_only=True):
    latent_size = latent.size()
    batch_size, channels, num_frames, height, width = mask.shape

    if process_first_frame_only:
        target_size = list(latent_size[2:])
        target_size[0] = 1
        first_frame_resized = F.interpolate(
            mask[:, :, 0:1, :, :],
            size=target_size,
            mode='trilinear',
            align_corners=False
        )

        target_size = list(latent_size[2:])
        target_size[0] = target_size[0] - 1
        if target_size[0] != 0:
            remaining_frames_resized = F.interpolate(
                mask[:, :, 1:, :, :],
                size=target_size,
                mode='trilinear',
                align_corners=False
            )
            resized_mask = torch.cat([first_frame_resized, remaining_frames_resized], dim=2)
        else:
            resized_mask = first_frame_resized
    else:
        target_size = list(latent_size[2:])
        resized_mask = F.interpolate(
            mask,
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
    return resized_mask


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def save_videos_from_pil(pil_images, path, fps=8):
    save_fmt = Path(path).suffix
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if save_fmt == ".mp4":
        with imageio.get_writer(path, fps=fps) as writer:
            for img in pil_images:
                img_array = np.array(img)  # Convert PIL Image to numpy array
                writer.append_data(img_array)

    elif save_fmt == ".gif":
        pil_images[0].save(
            fp=path,
            format="GIF",
            append_images=pil_images[1:],
            save_all=True,
            duration=(1 / fps * 1000),
            loop=0,
            optimize=False,
            lossless=True
        )
    else:
        raise ValueError("Unsupported file type. Use .mp4 or .gif.")


def save_videos_grid_png_and_mp4(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8, save_frames_path=None):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    height, width = videos.shape[-2:]
    outputs = []

    for i, x in enumerate(videos):
        x = torchvision.utils.make_grid(x, nrow=n_rows)  # (c h w)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)  # (h w c)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        x = Image.fromarray(x)
        outputs.append(x)

    pil_frames = [Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame for frame in outputs]
    num_frames = len(pil_frames)
    for i in range(num_frames):
        pil_frame = pil_frames[i]
        save_path = os.path.join(save_frames_path, f'frame_{i}.png')
        pil_frame.save(save_path)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    save_videos_from_pil(outputs, path, fps)


def log_validation(vae, text_encoder, tokenizer, clip_image_encoder, transformer3d, config, args, accelerator, weight_dtype,
                   global_step, height, width, wav2vec_processor, wav2vec):

    logger.info("Running validation... ")
    scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )
    pipeline = WanI2VFantasyPipeline(
        vae=accelerator.unwrap_model(vae).to(weight_dtype),
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        transformer=accelerator.unwrap_model(transformer3d),
        scheduler=scheduler,
        clip_image_encoder=accelerator.unwrap_model(clip_image_encoder),
    )
    pipeline = pipeline.to(accelerator.device)
    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    with torch.no_grad():
        with torch.autocast("cuda", dtype=weight_dtype):
            video_length = int((args.video_sample_n_frames - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if args.video_sample_n_frames != 1 else 1
            latent_frames = (video_length - 1) // vae.config.temporal_compression_ratio + 1

            torch.cuda.empty_cache()
            wav2vec.to(accelerator.device)
            fps = 30
            duration = librosa.get_duration(filename=args.validation_driven_audio_path)
            num_frames = min(int(fps * duration // 4) * 4 + 5, 81)
            if num_frames != 81:
                print(1/0)
            sr = 16000
            audio_input, sample_rate = librosa.load(args.validation_driven_audio_path, sr=sr)  # 采样率为 16kHz
            start_time = 0
            end_time = num_frames / fps
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            try:
                audio_segment = audio_input[start_sample:end_sample]
            except:
                audio_segment = audio_input
            vocal_input_values = wav2vec_processor(audio_segment, sampling_rate=sample_rate, return_tensors="pt").input_values.to(accelerator.device)
            with torch.no_grad():
                vocal_input_values = wav2vec(vocal_input_values).last_hidden_state

            wav2vec.to("cpu")
            torch.cuda.empty_cache()

            input_video, input_video_mask, clip_image = get_image_to_video_latent(args.validation_reference_path, None, video_length=video_length, sample_size=[height, width])

            text_prompt = "The protagonist is singing"
            num_inference_steps = 50
            sample = pipeline(
                text_prompt,
                num_frames=video_length,
                negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                height=height,
                width=width,
                guidance_scale=6.0,
                generator=generator,
                num_inference_steps=num_inference_steps,

                video=input_video,
                mask_video=input_video_mask,
                clip_image=clip_image,
                prompt_cfg_scale=5.0,
                audio_cfg_scale=5.0,
                vocal_input_values=vocal_input_values,
            ).videos
            os.makedirs(os.path.join(args.output_dir, "sample"), exist_ok=True)

            # save_frames_path = os.path.join(args.output_dir, f"sample/sample-{global_step}-frames")
            # os.makedirs(save_frames_path, exist_ok=True)
            fps = 16
            video_path = os.path.join(args.output_dir, f"sample/sample-{global_step}.mp4")
            # save_videos_grid_png_and_mp4(sample, os.path.join(args.output_dir, f"sample/sample-{global_step}.mp4"), n_rows=sample.shape[0], fps=fps, save_frames_path=save_frames_path)
            save_videos_grid(sample, video_path, fps=fps)

    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def linear_decay(initial_value, final_value, total_steps, current_step):
    if current_step >= total_steps:
        return final_value
    current_step = max(0, current_step)
    step_size = (final_value - initial_value) / total_steps
    current_value = initial_value + step_size * current_step
    return current_value


def generate_timestep_with_lognorm(low, high, shape, device="cpu", generator=None):
    u = torch.normal(mean=0.0, std=1.0, size=shape, device=device, generator=generator)
    t = 1 / (1 + torch.exp(-u)) * (high - low) + low
    return torch.clip(t.to(torch.int32), low, high - 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--use_came",
        action="store_true",
        help="whether to use came",
    )
    parser.add_argument(
        "--multi_stream",
        action="store_true",
        help="whether to use cuda multi-stream",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--vae_mini_batch", type=int, default=32, help="mini batch size for vae."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=2000,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    parser.add_argument(
        "--rank",
        type=int,
        default=128,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--network_alpha",
        type=int,
        default=64,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--snr_loss", action="store_true", help="Whether or not to use snr_loss."
    )
    parser.add_argument(
        "--uniform_sampling", action="store_true", help="Whether or not to use uniform_sampling."
    )
    parser.add_argument(
        "--enable_text_encoder_in_dataloader", action="store_true",
        help="Whether or not to use text encoder in dataloader."
    )
    parser.add_argument(
        "--random_ratio_crop", action="store_true", help="Whether enable random ratio crop sample in datasets."
    )
    parser.add_argument(
        "--random_frame_crop", action="store_true", help="Whether enable random frame crop sample in datasets."
    )
    parser.add_argument(
        "--random_hw_adapt", action="store_true", help="Whether enable random adapt height and width in datasets."
    )
    parser.add_argument(
        "--noise_share_in_frames", action="store_true", help="Whether enable noise share in frames."
    )
    parser.add_argument(
        "--noise_share_in_frames_ratio", type=float, default=0.5, help="Noise share ratio.",
    )
    parser.add_argument(
        "--motion_sub_loss", action="store_true", help="Whether enable motion sub loss."
    )
    parser.add_argument(
        "--motion_sub_loss_ratio", type=float, default=0.25, help="The ratio of motion sub loss."
    )
    parser.add_argument(
        "--keep_all_node_same_token_length",
        action="store_true",
        help="Reference of the length token.",
    )
    parser.add_argument(
        "--train_sampling_steps",
        type=int,
        default=1000,
        help="Run train_sampling_steps.",
    )
    parser.add_argument(
        "--video_sample_n_frames",
        type=int,
        default=17,
        help="Num frame of video.",
    )
    parser.add_argument(
        "--video_repeat",
        type=int,
        default=0,
        help="Num of repeat video.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help=(
            "The config of the model in training."
        ),
    )
    parser.add_argument(
        "--image_repeat_in_forward",
        type=int,
        default=0,
        help="Num of repeat image in forward.",
    )
    parser.add_argument(
        "--transformer_path",
        type=str,
        default=None,
        help=("If you want to load the weight from other transformers, input its path."),
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help=("If you want to load the weight from other vaes, input its path."),
    )
    parser.add_argument("--save_state", action="store_true", help="Whether or not to save state.")

    parser.add_argument(
        '--tokenizer_max_length',
        type=int,
        default=226,
        help='Max length of tokenizer'
    )
    parser.add_argument(
        "--use_deepspeed", action="store_true", help="Whether or not to use deepspeed."
    )
    parser.add_argument(
        "--low_vram", action="store_true", help="Whether enable low_vram mode."
    )
    parser.add_argument(
        "--train_mode",
        type=str,
        default="normal",
        help=(
            'The format of training data. Support `"normal"`'
            ' (default), `"inpaint"`.'
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--vocal_window_size",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--pretrained_wav2vec_path",
        type=str,
    )
    parser.add_argument(
        "--pretrained_musicfm_stat_path",
        type=str,
    )
    parser.add_argument(
        "--pretrained_musicfm_model_path",
        type=str,
    )
    parser.add_argument(
        "--validation_reference_path",
        type=str,
    )
    parser.add_argument(
        "--validation_driven_audio_path",
        type=str,
    )
    parser.add_argument(
        "--initial_grad_norm_ratio",
        type=int,
        default=5,
        help=(
            'The initial gradient is relative to the multiple of the max_grad_norm. '
        ),
    )
    parser.add_argument(
        "--abnormal_norm_clip_start",
        type=int,
        default=1000,
        help=(
            'When do we start doing additional processing on abnormal gradients. '
        ),
    )
    parser.add_argument(
        "--train_data_rec_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. "
        ),
    )
    parser.add_argument(
        "--train_data_square_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. "
        ),
    )
    parser.add_argument(
        "--train_data_vec_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. "
        ),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


#  This is for training using deepspeed.
#  Since now the DeepSpeed only supports trainging with only one model
#  So we create a virtual wrapper to contail all the models
class DeepSpeedWrapperModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        for name, value in kwargs.items():
            assert isinstance(value, nn.Module)
            self.register_module(name, value)


class CombinedModel(nn.Module):
    def __init__(self, transformer3d, lora_network, vocal_projector_model, audio_projector_model, motion_bucket_model):
        super().__init__()
        self.transformer3d = transformer3d
        self.lora_network = lora_network
        self.vocal_projector_model = vocal_projector_model
        self.audio_projector_model = audio_projector_model
        self.motion_bucket_model = motion_bucket_model

# Function for unwrapping if model was compiled with `torch.compile`.
def unwrap_model(accelerator, model):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    config = OmegaConf.load(args.config_path)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    deepspeed_plugin = accelerator.state.deepspeed_plugin
    if deepspeed_plugin is not None:
        zero_stage = int(deepspeed_plugin.zero_stage)
        print(f"Using DeepSpeed Zero stage: {zero_stage}")
    else:
        zero_stage = 0
        print("DeepSpeed is not enabled.")
    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=logging_dir)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)
        rng = np.random.default_rng(np.random.PCG64(args.seed + accelerator.process_index))
        torch_rng = torch.Generator(accelerator.device).manual_seed(args.seed + accelerator.process_index)
    else:
        rng = None
        torch_rng = None
    index_rng = np.random.default_rng(np.random.PCG64(43))
    print(
        f"Init rng with seed {args.seed + accelerator.process_index}. Process_index is {accelerator.process_index}")

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer3d) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Load scheduler, tokenizer and models.
    noise_scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )

    # Get Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.pretrained_model_name_or_path, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')), )

    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path,
                     config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
        additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )
    text_encoder = text_encoder.eval()
    # Get Vae
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
    )

    wav2vec_processor = Wav2Vec2Processor.from_pretrained(args.pretrained_wav2vec_path)
    wav2vec = Wav2Vec2Model.from_pretrained(args.pretrained_wav2vec_path).to("cpu")


    transformer3d = WanTransformer3DFantasyModel.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path,
                     config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
    ).to(weight_dtype)
    if args.transformer_path is not None:
        print(f"From checkpoint: {args.transformer_path}")
        state_dict = torch.load(args.transformer_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
        m, u = transformer3d.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

    # 14B should be audio_context_dim=2048
    # 1.3B should be audio_context_dim=1024
    # vocal_projector = FantasyTalkingVocalConditionModel(audio_in_dim=768, audio_proj_dim=2048)
    # vocal_projector = FantasyTalkingVocalConditionModel(audio_in_dim=768, audio_proj_dim=1024)

    clip_image_encoder = CLIPModel.from_pretrained(os.path.join(args.pretrained_model_name_or_path, config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')), )
    clip_image_encoder = clip_image_encoder.eval()

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    transformer3d.requires_grad_(False)
    clip_image_encoder.requires_grad_(False)
    wav2vec.requires_grad_(False)

    transformer3d.train()

    if args.gradient_checkpointing:
        transformer3d.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    elif args.use_came:
        try:
            from came_pytorch import CAME
        except:
            raise ImportError(
                "Please install came_pytorch to use CAME. You can do so by running `pip install came_pytorch`"
            )

        optimizer_cls = CAME
    else:
        optimizer_cls = torch.optim.AdamW

    trainable_params_optim = []
    for name, para in transformer3d.named_parameters():
        if "vocal_projector" in name or "audio_projector" in name or "vocal" in name or "audio" in name or "attn" in name or "blocks" in name:
            para.requires_grad = True
            trainable_params_optim.append({"params": para, "lr": args.learning_rate})
    # for name, para in vocal_projector.named_parameters():
    #     para.requires_grad = True
    #     trainable_params_optim.append({"params": para, "lr": args.learning_rate})

    trainable_params = list(filter(lambda p: p.requires_grad, transformer3d.parameters()))
    # trainable_params += list(filter(lambda p: p.requires_grad, vocal_projector.parameters()))

    if args.use_came:
        optimizer = optimizer_cls(
            trainable_params_optim,
            lr=args.learning_rate,
            # weight_decay=args.adam_weight_decay,
            betas=(0.9, 0.999, 0.9999),
            eps=(1e-30, 1e-16)
        )
    else:
        optimizer = optimizer_cls(
            trainable_params_optim,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    sample_n_frames_bucket_interval = vae.config.temporal_compression_ratio

    train_square_dataset = LargeScaleTalkingFantasyVideos(
        txt_path=args.train_data_square_dir,
        width=512,
        height=512,
        n_sample_frames=args.video_sample_n_frames,
        wav2vec_processor=wav2vec_processor,
        sample_frame_rate=1,
        only_last_features=False,
        vocal_sample_rate=16000,
        audio_sample_rate=24000,
        enable_inpaint=True,
        vae_stride=(4, 8, 8),
        patch_size=(1, 2, 2)
    )
    train_square_dataloader = torch.utils.data.DataLoader(
        train_square_dataset,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        shuffle=True,
    )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_square_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    transformer3d, optimizer, train_square_dataloader, lr_scheduler = accelerator.prepare(transformer3d, optimizer, train_square_dataloader, lr_scheduler)

    vae.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
    text_encoder.to(accelerator.device if not args.low_vram else "cpu")
    clip_image_encoder.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
    wav2vec.to(accelerator.device if not args.low_vram else "cpu")

    num_update_steps_per_epoch = math.ceil(len(train_square_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("Talking_Face", config=vars(args))


    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps


    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_square_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
            print(f"Resuming from checkpoint {path}. Get first_epoch = {first_epoch}.")
            accelerator.load_state(os.path.join(args.output_dir, path))
    else:
        initial_global_step = 0

    # function for saving/removing
    def save_model(ckpt_file, unwrapped_nw):
        os.makedirs(args.output_dir, exist_ok=True)
        accelerator.print(f"\nsaving checkpoint: {ckpt_file}")
        unwrapped_nw.save_weights(ckpt_file, weight_dtype, None)

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    if args.multi_stream and args.train_mode != "normal":
        # create extra cuda streams to speedup inpaint vae computation
        vae_stream_1 = torch.cuda.Stream()
        vae_stream_2 = torch.cuda.Stream()
    else:
        vae_stream_1 = None
        vae_stream_2 = None

    idx_sampling = DiscreteSampling(args.train_sampling_steps, uniform_sampling=args.uniform_sampling)

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0

        for step, batch in enumerate(train_square_dataloader):

            with accelerator.accumulate(transformer3d):
                pixel_values = batch["pixel_values"].to(weight_dtype)
                masked_pixel_values = batch["masked_pixel_values"].to(weight_dtype)
                pixel_value_masks = batch["pixel_value_masks"].to(weight_dtype)
                clip_pixel_values = batch["clip_pixel_values"].to(weight_dtype)
                vocal_input_values = batch["vocal_input_values"].squeeze(0)
                tgt_face_masks = batch["tgt_face_masks"].to(weight_dtype)
                tgt_lip_masks = batch["tgt_lip_masks"].to(weight_dtype)

                if args.train_mode != "normal":
                    t2v_flag = [(_mask == 1).all() for _mask in pixel_value_masks]
                    new_t2v_flag = []
                    for _mask in t2v_flag:
                        if _mask and np.random.rand() < 0.90:
                            new_t2v_flag.append(0)
                        else:
                            new_t2v_flag.append(1)
                    t2v_flag = torch.from_numpy(np.array(new_t2v_flag)).to(accelerator.device, dtype=weight_dtype)

                if args.low_vram:
                    torch.cuda.empty_cache()
                    vae.to(accelerator.device)
                    clip_image_encoder.to(accelerator.device)
                    text_encoder.to("cpu")
                    wav2vec.to("cpu")

                with torch.no_grad():
                    # This way is quicker when batch grows up
                    def _batch_encode_vae(pixel_values):
                        pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                        bs = args.vae_mini_batch
                        new_pixel_values = []
                        for i in range(0, pixel_values.shape[0], bs):
                            pixel_values_bs = pixel_values[i: i + bs]
                            pixel_values_bs = vae.encode(pixel_values_bs)[0]
                            pixel_values_bs = pixel_values_bs.sample()
                            new_pixel_values.append(pixel_values_bs)
                        return torch.cat(new_pixel_values, dim=0)

                    if vae_stream_1 is not None:
                        vae_stream_1.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(vae_stream_1):
                            latents = _batch_encode_vae(pixel_values)
                    else:
                        latents = _batch_encode_vae(pixel_values)

                    pixel_value_masks = rearrange(pixel_value_masks, "b f c h w -> b c f h w")
                    pixel_value_masks = torch.concat(
                        [
                            torch.repeat_interleave(pixel_value_masks[:, :, 0:1], repeats=4, dim=2),
                            pixel_value_masks[:, :, 1:]
                        ], dim=2
                    )
                    pixel_value_masks = pixel_value_masks.view(pixel_value_masks.shape[0], pixel_value_masks.shape[2] // 4, 4, pixel_value_masks.shape[3], pixel_value_masks.shape[4])
                    pixel_value_masks = pixel_value_masks.transpose(1, 2)
                    pixel_value_masks = resize_mask(1 - pixel_value_masks, latents)

                    masked_latents = _batch_encode_vae(masked_pixel_values)
                    if vae_stream_2 is not None:
                        torch.cuda.current_stream().wait_stream(vae_stream_2)

                    inpaint_latents = torch.concat([pixel_value_masks, masked_latents], dim=1)
                    inpaint_latents = t2v_flag[:, None, None, None, None] * inpaint_latents

                    clip_context = []
                    for clip_pixel_value in clip_pixel_values:
                        clip_image = Image.fromarray(np.uint8(clip_pixel_value.float().cpu().numpy()))
                        clip_image = TF.to_tensor(clip_image).sub_(0.5).div_(0.5).to(clip_image_encoder.device, weight_dtype)
                        _clip_context = clip_image_encoder([clip_image[:, None, :, :]])
                        clip_context.append(_clip_context)
                    clip_context = torch.cat(clip_context)

                if vae_stream_1 is not None:
                    torch.cuda.current_stream().wait_stream(vae_stream_1)

                if args.low_vram:
                    vae.to('cpu')
                    clip_image_encoder.to('cpu')
                    torch.cuda.empty_cache()
                    text_encoder.to(accelerator.device)

                with torch.no_grad():
                    prompt_ids = tokenizer(
                        batch['text_prompt'],
                        padding="max_length",
                        max_length=args.tokenizer_max_length,
                        truncation=True,
                        add_special_tokens=True,
                        return_tensors="pt"
                    )
                    text_input_ids = prompt_ids.input_ids
                    prompt_attention_mask = prompt_ids.attention_mask

                    seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
                    prompt_embeds = text_encoder(text_input_ids.to(latents.device), attention_mask=prompt_attention_mask.to(latents.device))[0]
                    prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]

                if args.low_vram:
                    text_encoder.to('cpu')
                    torch.cuda.empty_cache()
                    wav2vec.to(accelerator.device)

                with torch.no_grad():
                    try:
                        vocal_embeddings = wav2vec(vocal_input_values).last_hidden_state
                    except Exception as e:
                        audio_path = batch["audio_path"]
                        print(f"{audio_path} cannot be loaded !!!")
                        vocal_embeddings = torch.zeros([1, 168, 768])
                    if random.random() < 0.1:
                        vocal_embeddings = torch.zeros_like(vocal_embeddings)
                    if random.random() < 0.3:
                        is_clip_level_modeling = True
                    else:
                        is_clip_level_modeling = False

                if args.low_vram:
                    wav2vec.to('cpu')
                    torch.cuda.empty_cache()

                bsz, channel, num_frames, height, width = latents.size()
                noise = torch.randn(latents.size(), device=latents.device, generator=torch_rng, dtype=weight_dtype)

                if not args.uniform_sampling:
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme=args.weighting_scheme,
                        batch_size=bsz,
                        logit_mean=args.logit_mean,
                        logit_std=args.logit_std,
                        mode_scale=args.mode_scale,
                    )
                    indices = (u * noise_scheduler.config.num_train_timesteps).long()
                else:
                    # Sample a random timestep for each image
                    # timesteps = generate_timestep_with_lognorm(0, args.train_sampling_steps, (bsz,), device=latents.device, generator=torch_rng)
                    # timesteps = torch.randint(0, args.train_sampling_steps, (bsz,), device=latents.device, generator=torch_rng)
                    indices = idx_sampling(bsz, generator=torch_rng, device=latents.device)
                    indices = indices.long().cpu()
                timesteps = noise_scheduler.timesteps[indices].to(device=latents.device)

                def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
                    sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
                    schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
                    timesteps = timesteps.to(accelerator.device)
                    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

                    sigma = sigmas[step_indices].flatten()
                    while len(sigma.shape) < n_dim:
                        sigma = sigma.unsqueeze(-1)
                    return sigma

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
                noisy_latents = (1.0 - sigmas) * latents + sigmas * noise

                # Add noise
                target = noise - latents

                target_shape = (vae.latent_channels, num_frames, width, height)
                seq_len = math.ceil(
                    (target_shape[2] * target_shape[3]) /
                    (accelerator.unwrap_model(transformer3d).config.patch_size[1] *
                     accelerator.unwrap_model(transformer3d).config.patch_size[2]) *
                    target_shape[1]
                )

                # Predict the noise residual
                with torch.cuda.amp.autocast(dtype=weight_dtype):

                    noise_pred = transformer3d(
                        x=noisy_latents,
                        context=prompt_embeds,
                        t=timesteps,
                        seq_len=seq_len,
                        y=inpaint_latents,
                        clip_fea=clip_context,
                        vocal_embeddings=vocal_embeddings,
                        is_clip_level_modeling=is_clip_level_modeling,
                    )

                tgt_face_masks = F.interpolate(tgt_face_masks, size=(target.size()[-3], target.size()[-2], target.size()[-1]), mode='trilinear', align_corners=False)
                tgt_lip_masks = F.interpolate(tgt_lip_masks, size=(target.size()[-3], target.size()[-2], target.size()[-1]), mode='trilinear', align_corners=False)

                def custom_mse_loss(noise_pred, target, weighting=None, threshold=50, tgt_face_masks=None, tgt_lip_masks=None):
                    noise_pred = noise_pred.float()
                    target = target.float()
                    diff = noise_pred - target
                    mse_loss = F.mse_loss(noise_pred, target, reduction='none')

                    # training setting 1.0
                    mask_loss_flag = torch.rand(1).item()
                    if mask_loss_flag >= 0.4 and mask_loss_flag < 0.5:
                        mse_loss = mse_loss * tgt_face_masks
                    elif mask_loss_flag >= 0.5:
                        mse_loss = mse_loss * tgt_lip_masks
                    else:
                        mse_loss = mse_loss * (1 + tgt_face_masks + tgt_lip_masks)

                    if weighting is not None:
                        mse_loss = mse_loss * weighting
                    final_loss = mse_loss.mean()
                    return final_loss

                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                loss = custom_mse_loss(noise_pred.float(), target.float(), weighting.float(), tgt_face_masks=tgt_face_masks, tgt_lip_masks=tgt_lip_masks)
                loss = loss.mean()

                if args.motion_sub_loss and noise_pred.size()[1] > 2:
                    gt_sub_noise = noise_pred[:, 1:, :].float() - noise_pred[:, :-1, :].float()
                    pre_sub_noise = target[:, 1:, :].float() - target[:, :-1, :].float()
                    sub_loss = F.mse_loss(gt_sub_noise, pre_sub_noise, reduction="mean")
                    loss = loss * (1 - args.motion_sub_loss_ratio) + sub_loss * args.motion_sub_loss_ratio

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    if not args.use_deepspeed:
                        trainable_params_grads = [p.grad for p in trainable_params if p.grad is not None]
                        trainable_params_total_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in trainable_params_grads]), 2)
                        max_grad_norm = linear_decay(args.max_grad_norm * args.initial_grad_norm_ratio, args.max_grad_norm, args.abnormal_norm_clip_start, global_step)
                        if trainable_params_total_norm / max_grad_norm > 5 and global_step > args.abnormal_norm_clip_start:
                            actual_max_grad_norm = max_grad_norm / min((trainable_params_total_norm / max_grad_norm), 10)
                        else:
                            actual_max_grad_norm = max_grad_norm
                    else:
                        actual_max_grad_norm = args.max_grad_norm
                    norm_sum = accelerator.clip_grad_norm_(trainable_params, actual_max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                with torch.cuda.device(accelerator.device):
                    torch.cuda.empty_cache()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if args.use_deepspeed or accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        accelerator_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(accelerator_save_path, exist_ok=True)
                        accelerator.save_state(accelerator_save_path)
                        logger.info(f"Saved state to {accelerator_save_path}")

                        transformer3d_saved_path = os.path.join(args.output_dir, f"checkpoint-{global_step}", f"transformer3d-checkpoint-{global_step}.pt")
                        unwrap_transformer3d = accelerator.unwrap_model(transformer3d)
                        unwrap_transformer3d_state_dict = unwrap_transformer3d.state_dict()
                        torch.save(unwrap_transformer3d_state_dict, transformer3d_saved_path)
                        print(f"Saved transformer to {transformer3d_saved_path}")

                if accelerator.is_main_process:
                    if global_step % args.validation_steps == 0:
                        torch.cuda.empty_cache()
                        log_validation(vae=vae,
                                       text_encoder=text_encoder,
                                       tokenizer=tokenizer,
                                       clip_image_encoder=clip_image_encoder,
                                       transformer3d=transformer3d,
                                       config=config,
                                       args=args,
                                       accelerator=accelerator,
                                       weight_dtype=weight_dtype,
                                       global_step=global_step,
                                       height=512,
                                       width=512,
                                       wav2vec_processor=wav2vec_processor,
                                       wav2vec=wav2vec,
                                       )
                        torch.cuda.empty_cache()

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        os.makedirs(accelerator_save_path, exist_ok=True)
        accelerator.save_state(accelerator_save_path)
        logger.info(f"Saved state to {accelerator_save_path}")

        transformer3d_saved_path = os.path.join(args.output_dir, f"checkpoint-{global_step}", f"transformer3d-checkpoint-{global_step}.pt")
        unwrap_transformer3d = accelerator.unwrap_model(transformer3d)
        unwrap_transformer3d_state_dict = unwrap_transformer3d.state_dict()
        torch.save(unwrap_transformer3d_state_dict, transformer3d_saved_path)
        print(f"Saved transformer to {transformer3d_saved_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()