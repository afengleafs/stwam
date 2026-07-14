"""Ablation training/inference paths for STWAM (proprio dropout, k-draws,
pooled-adaLN connector).

Implements a *two-phase* coupled forward (video tower first, then the action
expert against the cached per-layer K/V).  Because information flows strictly
video -> action in ``coupled_forward`` (the video stream never attends the
action stream), the two-phase form is mathematically identical to the
interleaved original — verified bitwise by ``scripts/verify_ablation.py``.
The split is what makes the ablation factors cheap:

  * k-draws (ABC): one video forward serves ``k`` independent
    (noise, timestep) draws of the tiny action expert;
  * proprio dropout (LaWAM): per-sample masking of the proprio token
    (the last context column appended by ``build_context``);
  * pooled-adaLN (ABC): learnable queries pool the final-layer context-frame
    video features into a vector added to the action expert's adaLN
    conditioning.  Zero-init output => loading an old checkpoint starts as a
    no-op, same philosophy as the adapters' zero-init gates.

RNG draw order matches ``STWAMModel.training_loss`` exactly
(t_video -> noise_video -> t_action -> noise_action), so with
``k_draws=1, proprio_dropout=0, pooled off`` the loss is bit-identical to the
original under a fixed seed.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PooledAdaLNCond(nn.Module):
    """ABC-style connector: M learnable queries attention-pool the context-frame
    video features; the pooled vector is added to the action expert's adaLN
    conditioning (which otherwise only carries the flow timestep).

    Input features are the *final-layer, post-update_video* video stream over
    the clean context frames — available both from the training two-phase
    forward and from the inference prefill loop, so train/inference see the
    same tensor.
    """

    def __init__(self, video_dim: int, out_dim: int, num_queries: int = 8,
                 dim: int = 128, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.queries = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.proj_k = nn.Linear(video_dim, dim)
        self.proj_v = nn.Linear(video_dim, dim)
        self.norm = nn.RMSNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(num_queries * dim, 4 * out_dim), nn.SiLU(),
            nn.Linear(4 * out_dim, out_dim),
        )
        # zero-init the output => module starts as a no-op on old checkpoints
        nn.init.constant_(self.mlp[-1].weight, 0)
        nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, vfeat_ctx: torch.Tensor) -> torch.Tensor:
        """vfeat_ctx: [B, Tc, H, W, Dv] final-layer context features -> [B, out_dim]."""
        B = vfeat_ctx.shape[0]
        tokens = vfeat_ctx.reshape(B, -1, vfeat_ctx.shape[-1])          # [B,N,Dv]
        k = self.proj_k(tokens)
        v = self.proj_v(tokens)
        q = self.norm(self.queries).unsqueeze(0).expand(B, -1, -1)      # [B,M,dim]
        hd = self.dim // self.num_heads

        def split(x):
            return x.view(B, x.shape[1], self.num_heads, hd).transpose(1, 2)

        pooled = F.scaled_dot_product_attention(split(q), split(k), split(v))
        pooled = pooled.transpose(1, 2).reshape(B, -1)                  # [B,M*dim]
        return self.mlp(pooled)


def _video_phase(model, x_video_t: torch.Tensor, t: torch.Tensor,
                 ctx: torch.Tensor | None, ctx_mask: torch.Tensor | None,
                 num_ctx: int):
    """Video side of ``coupled_forward`` (same per-layer op order: block ->
    video_kv on context frames -> gated text update), returning the video
    v-prediction, the per-layer action-read K/V cache, and the final-layer
    features for pooled conditioning."""
    dit = model.video.dit
    xv = model.video.patchify(x_video_t)
    time_cond = model.video.get_time_cond(t)
    cache = []
    for L in range(model.num_layers):
        xv = dit.blocks[L](xv, time_cond, num_views=dit.num_views)
        cache.append(model.adapters[L].video_kv(xv[:, :num_ctx]))
        xv = model.adapters[L].update_video(xv, ctx, ctx_mask)
    v_video = model.video.head(xv, time_cond)
    return v_video, cache, xv


def _action_phase(model, x_a_t: torch.Tensor, t_a: torch.Tensor,
                  ctx: torch.Tensor | None, ctx_mask: torch.Tensor | None,
                  cache: list, pooled: torch.Tensor | None) -> torch.Tensor:
    """Action expert over (possibly k-expanded) chunk drafts against cached K/V."""
    xa = model.action.embed(x_a_t)
    ca = model.action.cond(t_a)
    if pooled is not None:
        ca = ca + pooled
    a_ctx = model.action.embed_context(ctx)
    for L in range(model.num_layers):
        xa = model.action.blocks[L](xa, ca, a_ctx, ctx_mask)
        xa = model.adapters[L].update_action(xa, *cache[L], model.action_times)
    return model.action.head_out(xa)


def _expand_k(x, k: int):
    if x is None or k == 1:
        return x
    return x.repeat_interleave(k, dim=0)


def training_loss_ablation(model, z: torch.Tensor, action_chunk: torch.Tensor,
                           ctx: torch.Tensor | None = None,
                           ctx_mask: torch.Tensor | None = None, *,
                           k_draws: int = 1, proprio_dropout: float = 0.0,
                           action_is_pad: torch.Tensor | None = None,
                           image_is_pad: torch.Tensor | None = None):
    """Two-phase equivalent of ``STWAMModel.training_loss`` with the three
    ablation factors.  With ``k_draws=1, proprio_dropout=0.0`` and no pooled
    module this reproduces the original loss bit-for-bit under a fixed seed.

    ``proprio_dropout`` masks the *last* context column per sample — valid only
    when the batch carries proprio (``build_context`` appends the state token
    last), which the LIBERO training batches always do.
    """
    B, T, H, W, C = z.shape
    nh = min(model.config.num_history, T)

    # --- RNG draws, in the original order -------------------------------
    t = model.diffusion.sample_t(B, T, device=z.device)
    t[:, :nh] = 0
    noise = torch.randn_like(z)
    x_video_t = model.diffusion.q_sample(z, t, noise)
    x_video_t[:, :nh] = z[:, :nh]
    ac = model.diffusion.alphas_cumprod[t.reshape(-1)].view(B, T, 1, 1, 1)
    target_v = ac.sqrt() * noise - (1 - ac).sqrt() * z

    action_rep = _expand_k(action_chunk, k_draws)
    Bk = action_rep.shape[0]
    t_a = torch.rand(Bk, device=z.device)
    noise_a = torch.randn_like(action_rep)
    x_a_t = torch.lerp(action_rep, noise_a, t_a.view(Bk, 1, 1))
    target_a = noise_a - action_rep

    if proprio_dropout > 0.0:
        assert ctx is not None and ctx_mask is not None, "proprio dropout needs a context mask"
        drop = torch.rand(B, device=z.device) < proprio_dropout
        ctx_mask = ctx_mask.clone()
        ctx_mask[drop, -1] = False

    # --- phase 1: video tower (once) -------------------------------------
    v_video, cache, xv_final = _video_phase(model, x_video_t, t, ctx, ctx_mask, nh)

    pooled = None
    if getattr(model, "pooled_cond", None) is not None:
        pooled = model.pooled_cond(xv_final[:, :nh])

    # --- phase 2: action expert (k draws share the video pass) -----------
    cache_k = [(_expand_k(k_, k_draws), _expand_k(v_, k_draws)) for k_, v_ in cache]
    v_action = _action_phase(model, x_a_t, t_a, _expand_k(ctx, k_draws),
                             _expand_k(ctx_mask, k_draws), cache_k,
                             _expand_k(pooled, k_draws))

    # --- losses (identical formulas) --------------------------------------
    v_err = (v_video - target_v).pow(2).mean(dim=(2, 3, 4))
    keep = torch.ones(B, T, device=z.device)
    keep[:, :nh] = 0
    if image_is_pad is not None:
        keep = keep * (~image_is_pad).float()
    loss_video = (v_err * keep).sum() / keep.sum().clamp(min=1)

    a_err = (v_action - target_a).pow(2).mean(dim=2)
    if action_is_pad is not None:
        keep_a = (~_expand_k(action_is_pad, k_draws)).float()
        loss_action = (a_err * keep_a).sum() / keep_a.sum().clamp(min=1)
    else:
        loss_action = a_err.mean()

    total = model.config.loss_lambda_video * loss_video + model.config.loss_lambda_action * loss_action
    return total, {"loss_video": float(loss_video.detach()),
                   "loss_action": float(loss_action.detach())}


@torch.no_grad()
def sample_actions_ablation(model, anchor_lat: torch.Tensor,
                            ctx: torch.Tensor | None = None,
                            ctx_mask: torch.Tensor | None = None,
                            num_steps: int | None = None) -> torch.Tensor:
    """``sample_actions`` with pooled-adaLN conditioning: the prefill loop also
    yields the final-layer features, whose pooled vector is added to the flow
    conditioning at every step.  Matches ``sample_actions`` exactly when the
    pooled module is absent or zero-init."""
    num_steps = num_steps or model.config.action_sampling_steps
    B, Tc = anchor_lat.shape[0], anchor_lat.shape[1]
    device = anchor_lat.device
    t_video = torch.zeros(B, Tc, dtype=torch.long, device=device)

    dit = model.video.dit
    xv = model.video.patchify(anchor_lat)
    time_cond = model.video.get_time_cond(t_video)
    cache = []
    for L in range(model.num_layers):
        xv = dit.blocks[L](xv, time_cond, num_views=dit.num_views)
        cache.append(model.adapters[L].video_kv(xv))
        xv = model.adapters[L].update_video(xv, ctx, ctx_mask)

    pooled = None
    if getattr(model, "pooled_cond", None) is not None:
        pooled = model.pooled_cond(xv[:, : min(model.config.num_history, Tc)])

    a_ctx = model.action.embed_context(ctx)
    x_a = torch.randn(B, model.config.chunk_size, model.config.action_dim, device=device)
    schedule = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    for i in range(num_steps):
        t_cur, t_next = schedule[i], schedule[i + 1]
        dt = (t_cur - t_next)
        ca = model.action.cond(torch.full((B,), float(t_cur), device=device))
        if pooled is not None:
            ca = ca + pooled
        xa = model.action.embed(x_a)
        for L in range(model.num_layers):
            xa = model.action.blocks[L](xa, ca, a_ctx, ctx_mask)
            xa = model.adapters[L].update_action(xa, *cache[L], model.action_times)
        v_a = model.action.head_out(xa)
        x_a = x_a - dt * v_a
    return x_a


def freeze_layerwise_read_path(model) -> list[str]:
    """Pooled-"only" mode: zero + freeze every adapter's action-read path so the
    action expert can only see the video tower through the pooled adaLN vector.

    NOTE: ``projV`` is intentionally left trainable — it is *shared* with the
    text->video update path (``update_video`` uses the same projection), so
    freezing it would also cripple the video side.  With ``gate_a`` pinned at
    zero the read path contributes nothing and receives no gradient anyway.
    """
    frozen = []
    for i, ad in enumerate(model.adapters):
        with torch.no_grad():
            ad.gate_a.zero_()
        for name, mod in (("gate_a", None), ("projA", ad.projA),
                          ("projA_out", ad.projA_out), ("read_attn", ad.read_attn)):
            if mod is None:
                ad.gate_a.requires_grad_(False)
            else:
                mod.requires_grad_(False)
            frozen.append(f"adapters.{i}.{name}")
    return frozen
