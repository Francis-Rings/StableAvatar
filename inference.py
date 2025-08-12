import argparse
import gc
import logging
import math
import os
import random
import shutil
import subprocess
from functools import partial

import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms.functional as TF
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import DDIMScheduler, FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (EMAModel,
                                      compute_density_for_timestep_sampling,
                                      compute_loss_weighting_for_sd3)
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import librosa
from pathlib import Path
import imageio
import torchvision
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import torch.distributed as dist

from wan.dist import set_multi_gpus_devices
from wan.distributed.fsdp import shard_model
from wan.models.cache_utils import get_teacache_coefficients
from wan.models.wan_fantasy_transformer3d_1B import WanTransformer3DFantasyModel
from wan.models.wan_text_encoder import WanT5EncoderModel
from wan.models.wan_vae import AutoencoderKLWan
from wan.models.wan_image_encoder import CLIPModel
from wan.pipeline.wan_inference_long_pipeline import WanI2VTalkingInferenceLongPipeline

from wan.utils.discrete_sampler import DiscreteSampling
from wan.utils.fp8_optimization import replace_parameters_by_name, convert_weight_dtype_wrapper, \
    convert_model_weight_to_float8
from wan.utils.utils import get_image_to_video_latent, save_videos_grid

def save_video_ffmpeg(gen_video_samples, save_path, vocal_audio_path, fps=25, quality=10):
    def save_video(frames, save_path, fps, quality=9, ffmpeg_params=None, saved_frames_dir=None):
        writer = imageio.get_writer(
            save_path, fps=fps, quality=quality, ffmpeg_params=ffmpeg_params
        )
        idx = 0
        for frame in tqdm(frames, desc="Saving video"):
            frame = np.array(frame)
            frame_path = os.path.join(saved_frames_dir, f"frame_{idx}.png")
            idx = idx + 1
            imageio.imwrite(frame_path, frame)
            writer.append_data(frame)
        writer.close()

    save_path_tmp = os.path.join(save_path, "video_without_audio.mp4")
    saved_frames_dir = os.path.join(save_path, "animated_images")
    os.makedirs(saved_frames_dir, exist_ok=True)

    # video_audio = (gen_video_samples + 1) / 2  # C T H W
    video_audio = (gen_video_samples / 2 + 0.5).clamp(0, 1)
    video_audio = video_audio.permute(1, 2, 3, 0).cpu().numpy()
    video_audio = np.clip(video_audio * 255, 0, 255).astype(np.uint8)  # to [0, 255]
    save_video(video_audio, save_path_tmp, fps=fps, quality=quality, saved_frames_dir=saved_frames_dir)

    # crop audio according to video length
    _, T, _, _ = gen_video_samples.shape
    duration = T / fps
    save_path_crop_audio = os.path.join(save_path, "cropped_audio.wav")
    final_command = [
        "ffmpeg",
        "-i",
        vocal_audio_path,
        "-t",
        f'{duration}',
        save_path_crop_audio,
    ]
    subprocess.run(final_command, check=True)


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


# Will error if the minimal version of diffusers is not installed. Remove at your own risks
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


def save_videos_grid_png_and_mp4(videos: torch.Tensor, rescale=False, n_rows=6, save_frames_path=None):
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
        "--transformer_path",
        type=str,
        default=None,
        help="The path to the Wan checkpoint directory.")
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
        "--output_dir",
        type=str,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default="The protagonist is singing",
        nargs="+",
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--pretrained_wav2vec_path",
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
        "--offload_model",
        action="store_true",
    )
    parser.add_argument(
        "--ulysses_degree",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--ring_degree",
        type=int,
        default=1,
        help="The size of the ring attention parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--fsdp_dit",
        action="store_true",
        help="Whether to use FSDP for DiT.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--motion_frame",
        type=int,
        default=25,
        help="Driven frame length used in the mode of long video genration.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_text_guide_scale",
        type=float,
        default=5.0,
        help="Classifier free guidance scale for text control.")
    parser.add_argument(
        "--sample_audio_guide_scale",
        type=float,
        default=4.0,
        help="Classifier free guidance scale for audio control.")
    parser.add_argument(
        "--overlap_window_length",
        type=int,
        default=10,)
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help=(
            "The config of the model in training."
        ),
    )
    parser.add_argument(
        "--enable_teacache",
        action="store_true",
    )
    parser.add_argument(
        "--teacache_threshold",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--num_skip_start_steps",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--teacache_offload",
        action="store_true",
    )
    parser.add_argument(
        "--GPU_memory_mode",
        type=str,
        default="model_full_load",
        help="[model_full_load, sequential_cpu_offload, model_cpu_offload_and_qfloat8, model_cpu_offload]"
    )
    parser.add_argument(
        "--clip_sample_n_frames",
        type=int,
        default=81,
    )

    args = parser.parse_args()
    # env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    # if env_local_rank != -1 and env_local_rank != args.local_rank:
    #     args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    # if args.non_ema_revision is None:
    #     args.non_ema_revision = args.revision

    return args


def main():
    args = parse_args()
    config = OmegaConf.load(args.config_path)

    if args.ulysses_degree > 1 or args.ring_degree > 1:
        rank = int(os.getenv("RANK", 0))
        world_size = int(os.getenv("WORLD_SIZE", 1))
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        import torch.distributed as dist
        try:
            import xfuser
            from xfuser.core.distributed import (get_sequence_parallel_rank,
                                                 get_sequence_parallel_world_size,
                                                 get_sp_group, get_world_group,
                                                 init_distributed_environment,
                                                 initialize_model_parallel)
            from xfuser.core.long_ctx_attention import xFuserLongContextAttention
        except Exception as ex:
            get_sequence_parallel_world_size = None
            get_sequence_parallel_rank = None
            xFuserLongContextAttention = None
            get_sp_group = None
            get_world_group = None
            init_distributed_environment = None
            initialize_model_parallel = None
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
        print('parallel inference enabled: ulysses_degree=%d ring_degree=%d rank=%d world_size=%d' % (args.ulysses_degree, args.ring_degree, dist.get_rank(), dist.get_world_size()))
        assert dist.get_world_size() == args.ring_degree * args.ulysses_degree, "number of GPUs(%d) should be equal to ring_degree * ulysses_degree." % dist.get_world_size()
        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
        initialize_model_parallel(sequence_parallel_degree=dist.get_world_size(),
                                  ring_degree=args.ring_degree,
                                  ulysses_degree=args.ulysses_degree)
        device = torch.device(f"cuda:{get_world_group().local_rank}")
        print('rank=%d device=%s' % (get_world_group().rank, str(device)))
    else:
        device = "cuda"

    # device = set_multi_gpus_devices(args.ulysses_degree, args.ring_degree)
    weight_dtype = torch.bfloat16
    sampler_name = "Flow"
    # GPU_memory_mode = "model_full_load"
    clip_sample_n_frames = args.clip_sample_n_frames
    fps = 25
    # fsdp_dit = False

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.pretrained_model_name_or_path, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')), )
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
        additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )
    text_encoder = text_encoder.eval()
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
    )
    wav2vec_processor = Wav2Vec2Processor.from_pretrained(args.pretrained_wav2vec_path)
    wav2vec = Wav2Vec2Model.from_pretrained(args.pretrained_wav2vec_path).to("cpu")
    clip_image_encoder = CLIPModel.from_pretrained(os.path.join(args.pretrained_model_name_or_path, config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')), )
    clip_image_encoder = clip_image_encoder.eval()
    transformer3d = WanTransformer3DFantasyModel.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        low_cpu_mem_usage=False,
        torch_dtype=weight_dtype,
    )
    if args.transformer_path is not None:
        print(f"From checkpoint: {args.transformer_path}")
        state_dict = torch.load(args.transformer_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
        m, u = transformer3d.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
    Choosen_Scheduler = scheduler_dict = {
        "Flow": FlowMatchEulerDiscreteScheduler,
    }[sampler_name]
    scheduler = Choosen_Scheduler(
        **filter_kwargs(Choosen_Scheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )
    pipeline = WanI2VTalkingInferenceLongPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        transformer=transformer3d,
        clip_image_encoder=clip_image_encoder,
        scheduler=scheduler,
        wav2vec_processor=wav2vec_processor,
        wav2vec=wav2vec,
    )
    if args.ulysses_degree > 1 or args.ring_degree > 1:
        transformer3d.enable_multi_gpus_inference()
        if args.fsdp_dit:
            shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
            pipeline.transformer = shard_fn(pipeline.transformer)

    if args.GPU_memory_mode == "sequential_cpu_offload":
        replace_parameters_by_name(transformer3d, ["modulation", ], device=device)
        transformer3d.freqs = transformer3d.freqs.to(device=device)
        pipeline.enable_sequential_cpu_offload(device=device)
    elif args.GPU_memory_mode == "model_cpu_offload_and_qfloat8":
        convert_model_weight_to_float8(transformer3d, exclude_module_name=["modulation", ])
        convert_weight_dtype_wrapper(transformer3d, weight_dtype)
        pipeline.enable_model_cpu_offload(device=device)
    elif args.GPU_memory_mode == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)
    else:
        pipeline.to(device=device)

    coefficients = get_teacache_coefficients(args.pretrained_model_name_or_path) if args.enable_teacache else None
    if coefficients is not None:
        print(f"Enable TeaCache with threshold {args.teacache_threshold} and skip the first {args.num_skip_start_steps} steps.")
        pipeline.transformer.enable_teacache(
            coefficients,
            args.sample_steps,
            args.teacache_threshold,
            num_skip_start_steps=args.num_skip_start_steps,
            offload=args.teacache_offload
        )

    generator = torch.Generator(device=device).manual_seed(args.seed)

    with torch.no_grad():
        video_length = int((clip_sample_n_frames - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if clip_sample_n_frames != 1 else 1
        input_video, input_video_mask, clip_image = get_image_to_video_latent(args.validation_reference_path, None, video_length=video_length, sample_size=[args.height, args.width])
        sr = 16000
        vocal_input, sample_rate = librosa.load(args.validation_driven_audio_path, sr=sr)
        sample = pipeline(
            args.validation_prompts,
            num_frames=video_length,
            negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            height=args.height,
            width=args.width,
            guidance_scale=6.0,
            generator=generator,
            num_inference_steps=args.sample_steps,

            video=input_video,
            mask_video=input_video_mask,
            clip_image=clip_image,
            text_guide_scale=args.sample_text_guide_scale,
            audio_guide_scale=args.sample_audio_guide_scale,
            vocal_input_values=vocal_input,
            motion_frame=args.motion_frame,
            fps=fps,
            sr=sr,
            cond_file_path=args.validation_reference_path,
            seed=args.seed,
            overlap_window_length=args.overlap_window_length,
        ).videos
        del pipeline
        saved_frames_dir = os.path.join(args.output_dir, "animated_images")
        os.makedirs(saved_frames_dir, exist_ok=True)
        video_path = os.path.join(args.output_dir, f"video_without_audio.mp4")

        if args.ulysses_degree * args.ring_degree > 1:
            import torch.distributed as dist
            if dist.get_rank() == 0:
                save_videos_grid(sample, video_path, fps=fps)
        else:
            save_videos_grid(sample, video_path, fps=fps)

if __name__ == "__main__":
    main()
