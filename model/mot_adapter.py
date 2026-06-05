"""Zero-init joint MoT adapter (branch B).

Inserted after each matched (video Block, action Block) pair.  It is the *only*
cross-expert coupling point: it projects both streams into a shared attention
space `Da`, runs one joint self-attention (so action tokens can read the video
world-model latents), injects language/state into the video stream via
cross-attention, then writes residual deltas back through **zero-initialized
gates**.

At initialization `gate_v == gate_a == 0`, so the adapter is a no-op and the
whole STWAM model is numerically identical to the pretrained video DiT (run with
its native action conditioning) plus an independent action expert.  This is the
Flamingo / LLaMA-Adapter gated-cross-attention pattern.
"""
from __future__ import annotations

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._layers import CrossAttention


class _JointSelfAttn(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.q_norm = nn.RMSNorm(self.head_dim, elementwise_affine=False)
        self.k_norm = nn.RMSNorm(self.head_dim, elementwise_affine=False)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        B, S, D = x.shape
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.out_proj(x.transpose(1, 2).reshape(B, S, D))


class JointMoTAdapter(nn.Module):
    def __init__(self, video_dim: int, action_dim: int, da: int = 384, num_heads: int = 6,
                 ctx_dim: int = 4096, video_reads_action: bool = False) -> None:
        super().__init__()
        self.da = da
        self.video_reads_action = video_reads_action
        self.projV = nn.Linear(video_dim, da)
        self.projA = nn.Linear(action_dim, da)
        self.norm_joint = nn.RMSNorm(da, elementwise_affine=False, eps=1e-6)
        self.joint_attn = _JointSelfAttn(da, num_heads)
        self.norm_xattn = nn.RMSNorm(da, elementwise_affine=False, eps=1e-6)
        self.cross_attn = CrossAttention(da, num_heads, kv_dim=da)
        self.text_proj = nn.Linear(ctx_dim, da)
        self.projV_out = nn.Linear(da, video_dim)
        self.projA_out = nn.Linear(da, action_dim)
        # zero-init gates -> adapter starts as a no-op
        self.gate_v = nn.Parameter(torch.zeros(video_dim))
        self.gate_a = nn.Parameter(torch.zeros(action_dim))

    def _build_mask(self, nv: int, na: int, device, dtype) -> torch.Tensor:
        total = nv + na
        keep = torch.ones(total, total, dtype=torch.bool, device=device)
        if not self.video_reads_action:
            keep[:nv, nv:] = False  # video queries do not attend action keys
        mask = torch.zeros(total, total, device=device, dtype=dtype)
        mask.masked_fill_(~keep, float("-inf"))
        return mask.unsqueeze(0).unsqueeze(0)  # [1,1,S,S]

    def forward(self, vfeat: torch.Tensor, afeat: torch.Tensor,
                ctx: torch.Tensor | None, ctx_mask: torch.Tensor | None):
        # vfeat: [B,T,H,W,Dv]; afeat: [B,chunk,Da_action]
        B, T, H, W, Dv = vfeat.shape
        nv = T * H * W
        na = afeat.shape[1]
        vp = self.projV(einops.rearrange(vfeat, "b t h w d -> b (t h w) d"))
        ap = self.projA(afeat)
        joint = torch.cat([vp, ap], dim=1)
        mask = self._build_mask(nv, na, joint.device, joint.dtype)
        joint = joint + self.joint_attn(self.norm_joint(joint), mask)
        # language/state -> video stream
        if ctx is not None:
            jv = joint[:, :nv]
            jv = jv + self.cross_attn(self.norm_xattn(jv), self.text_proj(ctx), ctx_mask)
            joint = torch.cat([jv, joint[:, nv:]], dim=1)
        dv = self.projV_out(joint[:, :nv])
        da = self.projA_out(joint[:, nv:])
        dv = einops.rearrange(dv, "b (t h w) d -> b t h w d", t=T, h=H, w=W)
        vfeat = vfeat + self.gate_v * dv
        afeat = afeat + self.gate_a * da
        return vfeat, afeat
