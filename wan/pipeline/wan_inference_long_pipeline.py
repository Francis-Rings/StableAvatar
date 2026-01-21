#!/usr/bin/env python3
import inspect
import math
import random
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import BaseOutput, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from einops import rearrange
from PIL import Image
from transformers import AutoTokenizer
from transformers import Wav2Vec2Model, Wav2Vec2Processor

from wan.models.vocal_projector_fantasy import split_audio_sequence, split_tensor_with_padding
from wan.models.wan_fantasy_transformer3d_1B import WanTransformer3DFantasyModel
from wan.models.wan_image_encoder import CLIPModel
from wan.models.wan_text_encoder import WanT5EncoderModel
from wan.models.wan_vae import AutoencoderKLWan
from wan.utils.color_correction import match_and_blend_colors

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        pass
        ```
"""

def enable_vram_management_recursively(model: torch.nn.Module, module_map: dict, module_config: dict, max_num_param=None, overflow_module_config: dict = None, total_num_param=0):
    for name, module in model.named_children():
        for source_module, target_module in module_map.items():
            if isinstance(module, source_module):
                num_param = sum(p.numel() for p in module.parameters())
                if max_num_param is not None and total_num_param + num_param > max_num_param:
                    module_config_ = overflow_module_config
                else:
                    module_config_ = module_config
                module_ = target_module(module, **module_config_)
                setattr(model, name, module_)
                total_num_param += num_param
                break
        else:
            total_num_param = enable_vram_management_recursively(module, module_map, module_config, max_num_param, overflow_module_config, total_num_param)
    return total_num_param

def enable_vram_management(model, module_map: dict, module_config: dict, max_num_param=None, overflow_module_config: dict = None):
    enable_vram_management_recursively(model, module_map, module_config, max_num_param, overflow_module_config, total_num_param=0)
    model.vram_management_enabled = True

def timestep_transform(
    t,
    shift=5.0,
    num_timesteps=1000,
):
    t = t / num_timesteps
    # shift the timestep based on ratio
    new_t = shift * t / (1 + (shift - 1) * t)
    new_t = new_t * num_timesteps
    return new_t


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
        scheduler,
        num_inference_steps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


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


@dataclass
class WanI2VPipelineTalkingInferenceLongOutput(BaseOutput):
    r"""
    Output class for CogVideo pipelines.

    Args:
        video (`torch.Tensor`, `np.ndarray`, or List[List[PIL.Image.Image]]):
            List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing
            denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape
            `(batch_size, num_frames, channels, height, width)`.
    """

    videos: torch.Tensor


class WanI2VTalkingInferenceLongPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-video generation using Wan.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    """

    _optional_components = []
    model_cpu_offload_seq = "text_encoder->clip_image_encoder->transformer->vae"

    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
    ]

    def __init__(
            self,
            tokenizer: AutoTokenizer,
            text_encoder: WanT5EncoderModel,
            vae: AutoencoderKLWan,
            transformer: WanTransformer3DFantasyModel,
            clip_image_encoder: CLIPModel,
            scheduler: FlowMatchEulerDiscreteScheduler,
            wav2vec_processor: Wav2Vec2Processor,
            wav2vec: Wav2Vec2Model,
    ):
        super().__init__()

        self.register_modules(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            clip_image_encoder=clip_image_encoder,
            scheduler=scheduler,
            wav2vec_processor=wav2vec_processor,
            wav2vec=wav2vec,
        )

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae.config.spacial_compression_ratio)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae.config.spacial_compression_ratio)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae.spacial_compression_ratio, do_normalize=False, do_binarize=True,
            do_convert_grayscale=True
        )



    def _get_t5_prompt_embeds(
            self,
            prompt: Union[str, List[str]] = None,
            num_videos_per_prompt: int = 1,
            max_sequence_length: int = 512,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1: -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
        prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=prompt_attention_mask.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return [u[:v] for u, v in zip(prompt_embeds, seq_lens)]

    def encode_prompt(
            self,
            prompt: Union[str, List[str]],
            negative_prompt: Optional[Union[str, List[str]]] = None,
            do_classifier_free_guidance: bool = True,
            num_videos_per_prompt: int = 1,
            prompt_embeds: Optional[torch.Tensor] = None,
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            max_sequence_length: int = 512,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    def prepare_latents(
            self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        shape = (
            batch_size,
            num_channels_latents,
            (num_frames - 1) // self.vae.config.temporal_compression_ratio + 1,
            height // self.vae.config.spacial_compression_ratio,
            width // self.vae.config.spacial_compression_ratio,
        )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        if hasattr(self.scheduler, "init_noise_sigma"):
            latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_mask_latents(
            self, mask, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance,
            noise_aug_strength
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision

        if mask is not None:
            mask = mask.to(device=device, dtype=self.vae.dtype)
            bs = 1
            new_mask = []
            for i in range(0, mask.shape[0], bs):
                mask_bs = mask[i: i + bs]
                mask_bs = self.vae.encode(mask_bs)[0]
                mask_bs = mask_bs.mode()
                new_mask.append(mask_bs)
            mask = torch.cat(new_mask, dim=0)
            # mask = mask * self.vae.config.scaling_factor

        if masked_image is not None:
            masked_image = masked_image.to(device=device, dtype=self.vae.dtype)
            bs = 1
            new_mask_pixel_values = []
            for i in range(0, masked_image.shape[0], bs):
                mask_pixel_values_bs = masked_image[i: i + bs]
                mask_pixel_values_bs = self.vae.encode(mask_pixel_values_bs)[0]
                mask_pixel_values_bs = mask_pixel_values_bs.mode()
                new_mask_pixel_values.append(mask_pixel_values_bs)
            masked_image_latents = torch.cat(new_mask_pixel_values, dim=0)
            # masked_image_latents = masked_image_latents * self.vae.config.scaling_factor
        else:
            masked_image_latents = None

        return mask, masked_image_latents

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        frames = self.vae.decode(latents.to(self.vae.dtype)).sample
        frames = frames.cpu()
        frames = (frames / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        frames = frames.float().numpy()
        return frames

    def decode_latents_audio_video(self, latents: torch.Tensor) -> torch.Tensor:
        frames = self.vae.decode(latents.to(self.vae.dtype)).sample
        # frames = (frames / 2 + 0.5).clamp(0, 1)
        frames = frames.cpu().float()
        return frames


    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # Copied from diffusers.pipelines.latte.pipeline_latte.LattePipeline.check_inputs
    def check_inputs(
            self,
            prompt,
            height,
            width,
            negative_prompt,
            callback_on_step_end_tensor_inputs,
            prompt_embeds=None,
            negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
                k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    def infer_add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        """
        compatible with diffusers add_noise()
        """
        timesteps = timesteps.float() / self.num_timesteps
        timesteps = timesteps.view(timesteps.shape + (1,) * (len(noise.shape)-1))
        return (1 - timesteps) * original_samples + timesteps * noise

    @property
    def guidance_scale(self):
        return self._guidance_scale

    # @property
    # def num_timesteps(self):
    #     return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
            self,
            prompt: Optional[Union[str, List[str]]] = None,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            height: int = 480,
            width: int = 720,
            video: Union[torch.FloatTensor] = None,
            mask_video: Union[torch.FloatTensor] = None,
            num_frames: int = 81,
            num_inference_steps: int = 50,
            timesteps: Optional[List[int]] = None,
            guidance_scale: float = 6,
            num_videos_per_prompt: int = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: str = "numpy",
            return_dict: bool = False,
            callback_on_step_end: Optional[
                Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
            ] = None,
            attention_kwargs: Optional[Dict[str, Any]] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            clip_image: Image = None,
            max_sequence_length: int = 512,
            text_guide_scale=None,
            audio_guide_scale=None,
            vocal_input_values=None,
            motion_frame=None,
            fps=None,
            sr=None,
            cond_file_path=None,
            seed=None,
            overlap_window_length=None,
            overlapping_weight_scheme="uniform",
            clip_length=81,  # Default from original code
    ) -> Union[WanI2VPipelineTalkingInferenceLongOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.
        Args:

        Examples:

        Returns:

        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs
        num_videos_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            callback_on_step_end_tensor_inputs,
            prompt_embeds,
            negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Default call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        weight_dtype = self.text_encoder.dtype

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if do_classifier_free_guidance:
            # prompt_embeds = negative_prompt_embeds + prompt_embeds + prompt_embeds
            prompt_embeds = negative_prompt_embeds + negative_prompt_embeds + prompt_embeds

        # clip_length is now a parameter with default value 81
        # Calculate frames_per_batch based on clip_length and temporal compression ratio
        frames_per_batch = (clip_length - 1) // self.vae.config.temporal_compression_ratio + 1
        audio_token_per_frame = int(sr / fps)
        max_audio_index = vocal_input_values.shape[0]
        total_frames = int(max_audio_index / audio_token_per_frame)
        if isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps, mu=1)
        else:
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)

        latent_channels = self.vae.config.latent_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            latent_channels,
            total_frames,
            height,
            width,
            weight_dtype,
            device,
            generator,
            latents,
        )
        infer_length = latents.size()[2]
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        latents_all = latents.clone()
        clip_image = cond_image = Image.open(cond_file_path).convert('RGB')
        cond_image = cond_image.resize([width, height])
        clip_image = clip_image.resize([width, height])
        clip_image = torch.from_numpy(np.array(clip_image)).permute(2, 0, 1)
        clip_image = clip_image / 255
        clip_image = (clip_image - 0.5) * 2  # C H W
        cond_image = torch.from_numpy(np.array(cond_image)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0)
        cond_image = cond_image / 255
        cond_image = (cond_image - 0.5) * 2  # normalization
        cond_image = cond_image.to(device)  # 1 C 1 H W

        clip_image = clip_image.to(device, weight_dtype)
        clip_context = self.clip_image_encoder([clip_image[:, None, :, :]])
        clip_context = (torch.cat([clip_context, clip_context, clip_context], dim=0) if do_classifier_free_guidance else clip_context)
        video_frames = torch.zeros(1, cond_image.shape[1], clip_length - cond_image.shape[2], height, width).to(device)
        padding_frames_pixels_values = torch.concat([cond_image, video_frames], dim=2).to(dtype=torch.float32)
        _, masked_video_latents = self.prepare_mask_latents(
            None,
            padding_frames_pixels_values,
            batch_size,
            height,
            width,
            weight_dtype,
            device,
            generator,
            do_classifier_free_guidance,
            noise_aug_strength=None,
        )  # [1, 16, 21, 64, 64]
        msk = torch.ones(1, clip_length, masked_video_latents.size()[-2], masked_video_latents.size()[-1], device=device)
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, masked_video_latents.size()[-2], masked_video_latents.size()[-1])
        msk = msk.transpose(1, 2).to(dtype=torch.float32)
        mask_input = torch.cat([msk] * 3) if do_classifier_free_guidance else msk
        masked_video_latents_input = (torch.cat([masked_video_latents] * 3) if do_classifier_free_guidance else masked_video_latents)
        y = torch.cat([mask_input.to(device), masked_video_latents_input.to(device)], dim=1).to(device, weight_dtype)

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                pred_latents = torch.zeros_like(latents_all, dtype=latents_all.dtype, )
                arrive_last_frame = False
                index_start = 0
                overlap_window_length = overlap_window_length  # [5, 7, 10] longer length --> higher quality
                index_end = index_start + frames_per_batch
                index_previous_end = index_end
                while index_end <= infer_length:
                    self.scheduler._step_index = None
                    idx_list = [ii % latents_all.shape[2] for ii in range(index_start, index_end)]

                    if index_end == infer_length:
                        idx_list_audio = [ii % max_audio_index for ii in range(index_start * 4 * audio_token_per_frame, max_audio_index)]
                    else:
                        # Use actual frame count for this batch instead of hardcoded clip_length
                        actual_frames = len(idx_list) * 4  # Each latent frame corresponds to 4 original frames
                        idx_list_audio = [ii % max_audio_index for ii in range(index_start * 4 * audio_token_per_frame, index_start * 4 * audio_token_per_frame + actual_frames * audio_token_per_frame)]

                    # idx_list_audio = [ii % max_audio_index for ii in range(index_start * 4 * audio_token_per_frame, index_end * 4 * audio_token_per_frame)]
                    latents = latents_all[:, :, idx_list].clone()
                    sub_vocal_input_values = vocal_input_values[idx_list_audio]
                    sub_vocal_input_values = self.wav2vec_processor(sub_vocal_input_values, sampling_rate=sr, return_tensors="pt").input_values.to(device)
                    sub_vocal_embeddings = self.wav2vec(sub_vocal_input_values).last_hidden_state
                    latent_model_input = torch.cat([latents] * 3) if do_classifier_free_guidance else latents
                    if hasattr(self.scheduler, "scale_model_input"):
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    timestep = t.expand(latent_model_input.shape[0])
                    target_shape = (self.vae.config.latent_channels, (num_frames - 1) // self.vae.config.temporal_compression_ratio + 1, width // self.vae.config.spacial_compression_ratio, height // self.vae.config.spacial_compression_ratio)
                    seq_len = math.ceil((target_shape[2] * target_shape[3]) / (self.transformer.config.patch_size[1] * self.transformer.config.patch_size[2]) * target_shape[1])
                    if text_guide_scale is not None and audio_guide_scale is not None:
                        sub_vocal_embeddings = torch.cat([torch.zeros_like(sub_vocal_embeddings), sub_vocal_embeddings, sub_vocal_embeddings], dim=0)
                    with torch.amp.autocast('cuda', dtype=weight_dtype):
                        legal_compressed_frames_num = latents.size()[2]
                        noise_pred = self.transformer(
                            x=latent_model_input,
                            context=prompt_embeds,
                            t=timestep,
                            seq_len=seq_len,
                            y=y[:, :, :legal_compressed_frames_num],
                            clip_fea=clip_context,
                            vocal_embeddings=sub_vocal_embeddings,
                            is_clip_level_modeling=False,
                            video_sample_n_frames=clip_length,  # Pass the actual clip_length
                        )
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_drop_audio, noise_pred_cond = noise_pred.chunk(3)
                        noise_pred = noise_pred_uncond + audio_guide_scale * (noise_pred_drop_audio - noise_pred_uncond) + text_guide_scale * (noise_pred_cond - noise_pred_drop_audio)
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                    torch.cuda.empty_cache()
                    if index_start != 0 and i != 0:
                        overlap_window_weight = torch.zeros(1, 1, overlap_window_length, 1, 1).to(device=latents.device, dtype=latents.dtype)
                        if overlapping_weight_scheme == "uniform":
                            for j in range(overlap_window_length):
                                overlap_window_weight[:, :, j] = j / (overlap_window_length-1)
                        elif overlapping_weight_scheme == "log":
                            init_weight = torch.linspace(0, 1, overlap_window_length)
                            init_weight = torch.log1p(init_weight * (torch.exp(torch.tensor(1.0)) - 1))
                            norm_weights = (init_weight - init_weight.min()) / (init_weight.max() - init_weight.min())
                            for j in range(overlap_window_length):
                                overlap_window_weight[:, :, j] = norm_weights[j]

                        overlap_idx_list_start = [ii % latents.shape[2] for ii in range(0, overlap_window_length)]
                        overlap_idx_list_end = [ii % latents_all.shape[2] for ii in range(index_previous_end-overlap_window_length, index_previous_end)]
                        latents[:, :, overlap_idx_list_start] = latents[:, :, overlap_idx_list_start] * overlap_window_weight + pred_latents[:, :, overlap_idx_list_end] * (1-overlap_window_weight)
                        latents = latents.to(torch.bfloat16)
                        for iii in range(legal_compressed_frames_num):
                            p = (index_start + iii) % pred_latents.shape[2]
                            pred_latents[:, :, p] = latents[:, :, iii]
                    else:
                        latents = latents.to(torch.bfloat16)
                        for iii in range(legal_compressed_frames_num):
                            p = (index_start + iii) % pred_latents.shape[2]
                            pred_latents[:, :, p] = latents[:, :, iii]
                    if arrive_last_frame:
                        break
                    if index_end != infer_length:
                        index_previous_end = index_end
                        index_start = index_start + (frames_per_batch-overlap_window_length)
                        if (index_start + frames_per_batch) < infer_length:
                            index_end = index_start + frames_per_batch
                        else:
                            index_end = infer_length
                            arrive_last_frame = True
                latents_all = pred_latents
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
        latents = latents_all.float()[:, :, :infer_length]
        torch.cuda.empty_cache()
        if output_type == "numpy":
            video = self.decode_latents(latents)
        elif not output_type == "latent":
            video = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents
        # Offload all models
        self.maybe_free_model_hooks()
        if not return_dict:
            video = torch.from_numpy(video)
        return WanI2VPipelineTalkingInferenceLongOutput(videos=video)

