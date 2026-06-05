"""Shared lightweight layers for STWAM's new (non-vendored) modules:
action expert and the zero-init joint MoT adapter.

Conventions match semantic-wm (`_swm/models/model.py`): RMSNorm, SwiGLU,
adaLN-Zero modulation, packed-qkv attention with RoPE.  These operate on 1D
token sequences `[B, S, D]` (action chunk / flattened video tokens), as opposed
to the vendored video DiT which keeps the `[B,T,H,W,D]` factorized layout.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # shift/scale: [B, D] (broadcast over sequence) or [B, S, D]
    if shift.dim() == 2:
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
    return x * (1 + scale) + shift


class SwiGLU(nn.Module):
    """Matches `_swm.models.model.SwiGLU` (2/3 hidden rule)."""

    def __init__(self, in_features: int, hidden_features: int | None = None,
                 out_features: int | None = None, bias: bool = True) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


def _build_rope_1d(seq_len: int, head_dim: int, device, dtype, base: float = 10_000.0):
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(half, device=device, dtype=torch.float32) / half))
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    theta = torch.outer(pos, inv_freq)  # [S, half]
    cos = torch.cos(theta).to(dtype)
    sin = torch.sin(theta).to(dtype)
    return cos, sin  # [S, half]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.view(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope_1d(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q,k: [B, H, S, Dh]; cos/sin: [S, Dh/2] -> repeat_interleave to [S, Dh]
    cos = torch.repeat_interleave(cos, 2, dim=-1).unsqueeze(0).unsqueeze(0)
    sin = torch.repeat_interleave(sin, 2, dim=-1).unsqueeze(0).unsqueeze(0)
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot


class SelfAttention1D(nn.Module):
    """Packed-qkv self-attention over a 1D sequence with RoPE + RMSNorm q/k."""

    def __init__(self, dim: int, num_heads: int, use_rope: bool = True) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_rope = use_rope
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.q_norm = nn.RMSNorm(self.head_dim, elementwise_affine=False)
        self.k_norm = nn.RMSNorm(self.head_dim, elementwise_affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.use_rope:
            cos, sin = _build_rope_1d(S, self.head_dim, x.device, x.dtype)
            q, k = apply_rope_1d(q, k, cos, sin)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, S, D)
        return self.out_proj(x)


class CrossAttention(nn.Module):
    """Cross-attention: queries from `x` [B,S,D], keys/values from `ctx` [B,L,D].

    `ctx_mask` (bool [B,L], True = keep) becomes an additive attention mask.
    """

    def __init__(self, dim: int, num_heads: int, kv_dim: int | None = None) -> None:
        super().__init__()
        assert dim % num_heads == 0
        kv_dim = kv_dim or dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(kv_dim, dim, bias=False)
        self.v = nn.Linear(kv_dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.q_norm = nn.RMSNorm(self.head_dim, elementwise_affine=False)
        self.k_norm = nn.RMSNorm(self.head_dim, elementwise_affine=False)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, ctx_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, S, D = x.shape
        L = ctx.shape[1]
        q = self.q(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(ctx).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(ctx).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        attn_mask = None
        if ctx_mask is not None:
            # [B, 1, 1, L] additive
            attn_mask = torch.zeros(B, 1, 1, L, device=x.device, dtype=q.dtype)
            attn_mask.masked_fill_(~ctx_mask.view(B, 1, 1, L), float("-inf"))
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x.transpose(1, 2).reshape(B, S, D)
        return self.out_proj(x)
