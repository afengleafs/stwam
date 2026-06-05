"""Lightweight 1D action expert (trained from scratch).

A small DiT over the action chunk `[B, chunk, action_dim]`:
  * adaLN-Zero conditioning on the (per-sample) diffusion timestep,
  * self-attention over the chunk (1D RoPE),
  * **native** cross-attention to the language/state context (direct policy
    conditioning — the action expert always sees the instruction),
  * SwiGLU feed-forward.

Its hidden width / head count are independent of the video expert: the only
cross-expert coupling happens in the zero-init joint MoT adapter, which projects
each stream into a shared attention space.  So `hidden` can be small
(FastWAM-style light action head, default 128).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ._layers import CrossAttention, SelfAttention1D, SwiGLU, modulate


def _timestep_embedding(t: torch.Tensor, dim: int = 256, max_period: int = 10000) -> torch.Tensor:
    import math
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class ActionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ctx_dim: int) -> None:
        super().__init__()
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6, bias=True))
        self.norm1 = nn.RMSNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm_x = nn.RMSNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.RMSNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention1D(dim, num_heads, use_rope=True)
        self.cross_attn = CrossAttention(dim, num_heads, kv_dim=dim)
        self.ffwd = SwiGLU(in_features=dim, hidden_features=dim * 4)

    def forward(self, x: torch.Tensor, c: torch.Tensor, ctx: torch.Tensor | None,
                ctx_mask: torch.Tensor | None) -> torch.Tensor:
        # c: [B, dim] (per-sample). adaLN -> 6 modulations.
        m = self.adaLN_modulation(c).chunk(6, dim=-1)  # each [B, dim]
        x = x + self.attn(modulate(self.norm1(x), m[0], m[1])) * m[2].unsqueeze(1)
        if ctx is not None:
            x = x + self.cross_attn(self.norm_x(x), ctx, ctx_mask)
        x = x + self.ffwd(modulate(self.norm2(x), m[3], m[4])) * m[5].unsqueeze(1)
        return x


class ActionExpert(nn.Module):
    """Denoises an action chunk; exposes block-wise API for the coupled MoT loop."""

    def __init__(self, action_dim: int, dim: int = 128, num_layers: int = 12,
                 num_heads: int = 4, ctx_dim: int = 4096, time_freq_dim: int = 256) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.dim = dim
        self.num_layers = num_layers
        self.action_in_proj = nn.Linear(action_dim, dim)
        self.timestep_mlp = nn.Sequential(
            nn.Linear(time_freq_dim, dim, bias=True), nn.SiLU(), nn.Linear(dim, dim, bias=True))
        self.text_embedding = nn.Sequential(
            nn.Linear(ctx_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([ActionBlock(dim, num_heads, dim) for _ in range(num_layers)])
        self.head_norm = nn.RMSNorm(dim, elementwise_affine=False, eps=1e-6)
        self.head = nn.Linear(dim, action_dim)
        self._time_freq_dim = time_freq_dim
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for blk in self.blocks:
            nn.init.constant_(blk.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(blk.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.head.weight, 0)
        nn.init.constant_(self.head.bias, 0)

    def cond(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B] per-sample timestep -> c: [B, dim]
        return self.timestep_mlp(_timestep_embedding(t, self._time_freq_dim))

    def embed_context(self, ctx: torch.Tensor | None) -> torch.Tensor | None:
        return None if ctx is None else self.text_embedding(ctx)

    def embed(self, x_action: torch.Tensor) -> torch.Tensor:
        return self.action_in_proj(x_action)

    def head_out(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(self.head_norm(tokens))
