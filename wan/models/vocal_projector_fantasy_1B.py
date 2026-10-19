#!/usr/bin/env python3
import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from wan.models.vocal_projector_fantasy import split_audio_sequence, split_tensor_with_padding

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if torch.backends.cuda.flash_sdp_enabled() is False or torch.backends.cuda.enable_flash_sdp is False:
            print(1/0)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
    return out


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)


class VocalCrossAttention(nn.Module):
    def __init__(self,
                 vocal_dim,
                 dit_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6,

                 ):
        assert vocal_dim % num_heads == 0
        super().__init__()
        self.vocal_dim = vocal_dim
        self.dit_dim = dit_dim
        self.num_heads = num_heads
        self.head_dim = vocal_dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(vocal_dim, vocal_dim)
        self.k = nn.Linear(dit_dim, vocal_dim)
        self.v = nn.Linear(dit_dim, vocal_dim)
        self.o = nn.Linear(vocal_dim, vocal_dim)
        self.norm_q = WanRMSNorm(vocal_dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(vocal_dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, q_lens, dtype, latents_num_frames=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
            latents_num_frames: Number of latent frames (optional)
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim
        
        # Use provided latents_num_frames or default to 21
        if latents_num_frames is None:
            latents_num_frames = 21
            
        q = self.norm_q(self.q(x.to(dtype))).view(b * latents_num_frames, -1, n, d)
        k = self.norm_k(self.k(context.to(dtype))).view(b * latents_num_frames, -1, n, d)
        v = self.v(context.to(dtype)).view(b * latents_num_frames, -1, n, d)
        # compute attention

        x = attention(
            q.to(dtype),
            k.to(dtype),
            v.to(dtype),
            q_lens=None,
            k_lens=None,
        )
        x = x.to(dtype)
        x = x.view(b, -1, n, d)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class VocalAttentionBlock(nn.Module):

    def __init__(self,
                 vocal_dim,
                 dit_dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        super().__init__()
        self.vocal_dim = vocal_dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(vocal_dim, eps)
        self.norm3 = WanLayerNorm(
            vocal_dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()

        self.cross_attn = VocalCrossAttention(vocal_dim=vocal_dim,
                                              dit_dim=dit_dim,
                                              num_heads=num_heads,
                                              window_size=(-1, -1),
                                              qk_norm=qk_norm,
                                              eps=eps)
        self.norm2 = WanLayerNorm(vocal_dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(vocal_dim, ffn_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, vocal_dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, vocal_dim) / vocal_dim ** 0.5)

    def forward(
            self,
            x,
            e,
            context,
            q_lens,
            dtype=torch.float32,
            latents_num_frames=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        e = (self.modulation + e).chunk(6, dim=1)

        # self-attention
        if len(x.shape) == 4:
            b, t, n, d = x.size()
            x = rearrange(x, "b t n d -> b (t n) d", t=t)

        temp_x = self.norm1(x) * (1 + e[1]) + e[0]
        temp_x = temp_x.to(dtype)
        x = x + temp_x * e[2]

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, q_lens, e):
            # cross-attention
            x = x + self.cross_attn(self.norm3(x), context, q_lens, dtype, latents_num_frames)
            # ffn function
            temp_x = self.norm2(x) * (1 + e[4]) + e[3]
            temp_x = temp_x.to(dtype)

            y = self.ffn(temp_x)
            x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context, q_lens, e)
        return x


class Final_Head(nn.Module):

    def __init__(self, dim, out_dim, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps

        # layers
        self.norm = WanLayerNorm(dim, eps)
        self.final_proj = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
        x = (self.final_proj(self.norm(x) * (1 + e[1]) + e[0]))
        return x

class VocalProjModel(nn.Module):
    def __init__(self, audio_in_dim=1024, cross_attention_dim=1024):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.proj = torch.nn.Linear(audio_in_dim, cross_attention_dim, bias=False)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, audio_embeds):
        context_tokens = self.proj(audio_embeds)
        context_tokens = self.norm(context_tokens)
        return context_tokens  # [B,L,C]


class FantasyTalkingVocalCondition1BModel(nn.Module):
    def __init__(self, audio_in_dim: int, audio_proj_dim: int, dit_dim: int):
        super().__init__()

        self.audio_in_dim = audio_in_dim
        self.audio_proj_dim = audio_proj_dim
        # audio proj model
        self.proj_model = self.init_proj(self.audio_proj_dim)

        num_layers = 2
        self.blocks = nn.ModuleList([
            VocalAttentionBlock(
                vocal_dim=audio_proj_dim,
                dit_dim=dit_dim,
                ffn_dim=audio_proj_dim * 2,
                num_heads=8,
                window_size=(-1, -1),
                qk_norm=True,
                cross_attn_norm=True,
            )
            for _ in range(num_layers)
        ])

        self.final_head = Final_Head(dim=audio_proj_dim, out_dim=audio_proj_dim)

    def init_proj(self, cross_attention_dim=5120):
        proj_model = VocalProjModel(
            audio_in_dim=self.audio_in_dim, cross_attention_dim=cross_attention_dim
        )
        return proj_model

    def forward(self, vocal_embeddings=None, video_sample_n_frames=81, latents=None, e0=None, e=None):
        vocal_proj_feature = self.proj_model(vocal_embeddings)
        pos_idx_ranges = split_audio_sequence(vocal_proj_feature.size(1), num_frames=video_sample_n_frames)
        vocal_proj_split, vocal_context_lens = split_tensor_with_padding(vocal_proj_feature, pos_idx_ranges, expand_length=4)
        latents_num_frames = vocal_proj_split.size()[1]
        for block in self.blocks:
            vocal_proj_split = block(
                x=vocal_proj_split,
                e=e0,
                context=latents,
                q_lens=vocal_context_lens,
                latents_num_frames=latents_num_frames,
            )
        context_tokens = self.final_head(vocal_proj_split, e)
        context_tokens = rearrange(context_tokens, "b (f n) c -> b f n c", f=latents_num_frames)
        if vocal_embeddings.size()[0] > 1:
            vocal_context_lens = torch.cat([vocal_context_lens] * 3)
        return context_tokens, vocal_context_lens


if __name__ == "__main__":
    model = FantasyTalkingVocalCondition1BModel(audio_in_dim=768, audio_proj_dim=1536, dit_dim=1536)
    vocal_embeddings = torch.randn(3, 134, 768)
    latents = torch.randn(3, 21504, 1536)
    e0 = torch.randn(3, 6, 1536)
    e = torch.randn(3, 1536)
    out, seq_len = model(vocal_embeddings=vocal_embeddings, video_sample_n_frames=81, latents=latents, e0=e0, e=e)
    print(out.size())
    print(seq_len.size())