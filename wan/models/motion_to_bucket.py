#!/usr/bin/env python3
import torch
from diffusers import ModelMixin
from einops import rearrange
from torch import nn


class Motion2bucketModel(ModelMixin):
    def __init__(self, window_size=5, blocks=12, channels=1024, clip_channels=1280, intermediate_dim=512, output_dim=768, context_tokens=32, clip_token_num=1, final_output_dim=5120):
        super().__init__()
        self.window_size = window_size
        self.clip_token_num = clip_token_num
        self.blocks = blocks
        self.channels = channels
        # self.input_dim = (window_size * blocks * channels + clip_channels*clip_token_num)
        self.input_dim = (window_size * channels + clip_channels * clip_token_num)
        self.intermediate_dim = intermediate_dim
        self.context_tokens = context_tokens
        self.output_dim = output_dim

        # define multiple linear layers
        self.proj1 = nn.Linear(self.input_dim, intermediate_dim)
        self.proj2 = nn.Linear(intermediate_dim, intermediate_dim)
        self.proj3 = nn.Linear(intermediate_dim, context_tokens * output_dim)
        self.act = nn.SiLU()


        self.final_proj = torch.nn.Linear(output_dim, final_output_dim)
        self.final_norm = torch.nn.LayerNorm(final_output_dim)

        nn.init.constant_(self.final_proj.weight, 0)
        if self.final_proj.bias is not None:
            nn.init.constant_(self.final_proj.bias, 0)

    def forward(self, audio_embeds, clip_embeds):
        """
        Defines the forward pass for the AudioProjModel.

        Parameters:
            audio_embeds (torch.Tensor): The input audio embeddings with shape (batch_size, video_length, blocks, channels).

        Returns:
            context_tokens (torch.Tensor): The output context tokens with shape (batch_size, video_length, context_tokens, output_dim).
        """
        # merge
        video_length = audio_embeds.shape[1]
        # audio_embeds = rearrange(audio_embeds, "bz f w b c -> (bz f) w b c")
        audio_embeds = rearrange(audio_embeds, "bz f w c -> (bz f) w c")
        clip_embeds = clip_embeds.repeat(audio_embeds.size()[0]//clip_embeds.size()[0], 1, 1)
        clip_embeds = rearrange(clip_embeds, "b n d -> b (n d)")
        # batch_size, window_size, blocks, channels = audio_embeds.shape
        # audio_embeds = audio_embeds.view(batch_size, window_size * blocks * channels)
        batch_size, window_size, channels = audio_embeds.shape
        audio_embeds = audio_embeds.view(batch_size, window_size * channels)
        audio_embeds = torch.cat([audio_embeds, clip_embeds], dim=-1)

        audio_embeds = self.act(self.proj1(audio_embeds))
        audio_embeds = self.act(self.proj2(audio_embeds))

        context_tokens = self.proj3(audio_embeds).reshape(
            batch_size, self.context_tokens, self.output_dim
        )

        # context_tokens = self.norm(context_tokens)
        context_tokens = rearrange(
            context_tokens, "(bz f) m c -> bz f m c", f=video_length
        )

        context_tokens = self.act(context_tokens)
        context_tokens = self.final_norm(self.final_proj(context_tokens))

        return context_tokens


if __name__ == '__main__':
    model = Motion2bucketModel(window_size=5)
    # audio_features = torch.randn(1, 81, 5, 12, 768)
    audio_features = torch.randn(1, 81, 5, 1024)
    clip_image_features = torch.randn(1, 1, 1280)

    out = model(audio_features, clip_image_features).mean(dim=2).mean(dim=1)
    print(out.size())
