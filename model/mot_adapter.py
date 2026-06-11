"""Zero-init MoT adapter (branch B, FastWAM-aligned coupling).

Inserted after each matched (video Block, action Block) pair.  Two gated
one-way paths per layer:

  * action -> video (``_ActionReadAttn``): action queries cross-attend the
    *clean context* video tokens (the first ``num_ctx_frames`` history frames
    only -- never noisy future frames), with time-aligned RoPE: video keys get
    (frame, row, col) banded RoPE, action queries sit on the same time axis at
    their chunk positions.  This is the only channel through which the policy
    reads the world model, and it mirrors FastWAM's "action attends the clean
    conditioning frame" mask.
  * text -> video (``CrossAttention``): language/state injection into the
    video stream.

There is intentionally NO video<->video joint attention (it would duplicate
the DiT's own attention and break its temporal causality) and NO video->action
path (so the video tower is action-independent and its per-layer K/V can be
prefilled once and cached during action sampling).

At initialization ``gate_v == gate_a == 0``, so the adapter is a no-op and the
whole STWAM model is numerically identical to the pretrained video DiT (run
with its native action conditioning) plus an independent action expert.  This
is the Flamingo / LLaMA-Adapter gated-cross-attention pattern.
"""
from __future__ import annotations

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._layers import CrossAttention, _rotate_half


def _axis_cos_sin(pos: torch.Tensor, dim: int, device, dtype, base: float = 10_000.0):
    # pos: [S] (float ok) -> cos/sin [S, dim//2]
    half = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(half, device=device, dtype=torch.float32) / half))
    theta = torch.outer(pos.to(device=device, dtype=torch.float32), inv_freq)
    return torch.cos(theta).to(dtype), torch.sin(theta).to(dtype)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, S, Dh]; cos/sin: [S, Dh/2]
    cos = torch.repeat_interleave(cos, 2, dim=-1).unsqueeze(0).unsqueeze(0)
    sin = torch.repeat_interleave(sin, 2, dim=-1).unsqueeze(0).unsqueeze(0)
    return x * cos + _rotate_half(x) * sin


class _ActionReadAttn(nn.Module):
    """Cross-attention where action tokens read context video tokens.

    head_dim is split into (time, row, col) RoPE bands shared by q and k; the
    action query has no spatial coordinate, so its row/col bands rotate by 0.
    """

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        spatial = (self.head_dim // 3) // 2 * 2
        self.axis_dims = (self.head_dim - 2 * spatial, spatial, spatial)  # (t, h, w)
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.q_norm = nn.RMSNorm(self.head_dim, elementwise_affine=False)
        self.k_norm = nn.RMSNorm(self.head_dim, elementwise_affine=False)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        return x.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

    def kv(self, vp: torch.Tensor, T: int, H: int, W: int):
        """vp: [B, T*H*W, dim] (normed context video tokens) -> RoPE'd (k, v)."""
        k, v = self._heads(self.k(vp)), self._heads(self.v(vp))
        k = self.k_norm(k)
        dt, dh, dw = self.axis_dims
        dev, dtype = k.device, k.dtype
        t_pos = torch.arange(T, device=dev).repeat_interleave(H * W)
        h_pos = torch.arange(H, device=dev).repeat_interleave(W).repeat(T)
        w_pos = torch.arange(W, device=dev).repeat(T * H)
        cos_t, sin_t = _axis_cos_sin(t_pos, dt, dev, dtype)
        cos_h, sin_h = _axis_cos_sin(h_pos, dh, dev, dtype)
        cos_w, sin_w = _axis_cos_sin(w_pos, dw, dev, dtype)
        k = _apply_rope(k, torch.cat([cos_t, cos_h, cos_w], dim=-1),
                        torch.cat([sin_t, sin_h, sin_w], dim=-1))
        return k, v

    def attend(self, aq: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
               action_times: torch.Tensor) -> torch.Tensor:
        """aq: [B, Sa, dim] (normed action queries); action_times: [Sa] float."""
        B, Sa, _ = aq.shape
        q = self.q_norm(self._heads(self.q(aq)))
        dt, dh, dw = self.axis_dims
        cos_t, sin_t = _axis_cos_sin(action_times, dt, q.device, q.dtype)
        pad_c = torch.ones(Sa, (dh + dw) // 2, device=q.device, dtype=q.dtype)
        pad_s = torch.zeros_like(pad_c)
        q = _apply_rope(q, torch.cat([cos_t, pad_c], dim=-1), torch.cat([sin_t, pad_s], dim=-1))
        x = F.scaled_dot_product_attention(q, k, v)
        return self.out_proj(x.transpose(1, 2).reshape(B, Sa, -1))


class JointMoTAdapter(nn.Module):
    def __init__(self, video_dim: int, action_dim: int, da: int = 384, num_heads: int = 6,
                 ctx_dim: int = 4096) -> None:
        super().__init__()
        self.da = da
        # action -> video read path
        self.projV = nn.Linear(video_dim, da)
        self.projA = nn.Linear(action_dim, da)
        self.norm_v = nn.RMSNorm(da, elementwise_affine=False, eps=1e-6)
        self.norm_a = nn.RMSNorm(da, elementwise_affine=False, eps=1e-6)
        self.read_attn = _ActionReadAttn(da, num_heads)
        self.projA_out = nn.Linear(da, action_dim)
        # text/state -> video path
        self.norm_xattn = nn.RMSNorm(da, elementwise_affine=False, eps=1e-6)
        self.cross_attn = CrossAttention(da, num_heads, kv_dim=da)
        self.text_proj = nn.Linear(ctx_dim, da)
        self.projV_out = nn.Linear(da, video_dim)
        # zero-init gates -> adapter starts as a no-op
        self.gate_v = nn.Parameter(torch.zeros(video_dim))
        self.gate_a = nn.Parameter(torch.zeros(action_dim))

    def video_kv(self, vfeat_ctx: torch.Tensor):
        """Per-layer K/V over the context frames; cacheable across flow steps.

        vfeat_ctx: [B, Tc, H, W, Dv] -- pass *only* the clean context frames.
        """
        B, Tc, H, W, _ = vfeat_ctx.shape
        vp = self.norm_v(self.projV(einops.rearrange(vfeat_ctx, "b t h w d -> b (t h w) d")))
        return self.read_attn.kv(vp, Tc, H, W)

    def update_action(self, afeat: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                      action_times: torch.Tensor) -> torch.Tensor:
        aq = self.norm_a(self.projA(afeat))
        delta = self.projA_out(self.read_attn.attend(aq, k, v, action_times))
        return afeat + self.gate_a * delta

    def update_video(self, vfeat: torch.Tensor, ctx: torch.Tensor | None,
                     ctx_mask: torch.Tensor | None) -> torch.Tensor:
        if ctx is None:
            return vfeat
        B, T, H, W, _ = vfeat.shape
        vp = self.projV(einops.rearrange(vfeat, "b t h w d -> b (t h w) d"))
        delta = self.projV_out(self.cross_attn(self.norm_xattn(vp), self.text_proj(ctx), ctx_mask))
        delta = einops.rearrange(delta, "b (t h w) d -> b t h w d", t=T, h=H, w=W)
        return vfeat + self.gate_v * delta

    def forward(self, vfeat: torch.Tensor, afeat: torch.Tensor,
                ctx: torch.Tensor | None, ctx_mask: torch.Tensor | None,
                num_ctx_frames: int, action_times: torch.Tensor):
        # vfeat: [B,T,H,W,Dv]; afeat: [B,chunk,Da_action]
        k, v = self.video_kv(vfeat[:, :num_ctx_frames])
        afeat = self.update_action(afeat, k, v, action_times)
        vfeat = self.update_video(vfeat, ctx, ctx_mask)
        return vfeat, afeat
