"""Inference-time future rollout for STWAM (FastWAM-IDM-style diagnostic).

Adds a second inference path next to ``STWAMModel.sample_actions``:

  1. ``rollout_future_latents`` — diffusion-forcing DDIM rollout of the
     ``n_frames - num_history`` future semantic-latent frames through the
     coupled model's *video side only* (history frames stay teacher-forced
     clean at t=0, exactly the finetuning convention of ``training_loss``);
  2. ``prefill_video_window`` — the standard prefill over the full clean
     window, with an explicit ``num_ctx_frames`` knob for how many frames the
     action expert may read;
  3. ``sample_actions_rollout`` — rollout + prefill + the unchanged
     flow-matching action loop.

Everything here is additive: no trained module is modified.  The DDIM update
is inlined (v-pred -> x0 -> eps -> x_next, same algebra as
``_swm/training/diffusion.py`` ``ddim_sample_step``) rather than reusing
``Diffusion.generate``, because that path drives the bare DiT with its own
action conditioning (never used by STWAM) and a different clean-context
convention (``stabilization_level`` instead of exact t=0 latents).

Sampler state is kept in fp32 even under bf16 autocast; only the network
forwards run in the ambient autocast dtype.
"""
from __future__ import annotations

import torch

from .modeling_stwam import STWAMModel


@torch.no_grad()
def video_only_forward(model: STWAMModel, x_lat: torch.Tensor, t: torch.Tensor,
                       ctx: torch.Tensor | None, ctx_mask: torch.Tensor | None) -> torch.Tensor:
    """Video side of ``coupled_forward`` (blocks + adapter text update + head).

    x_lat: [B,T,H,W,C] latents; t: [B,T] long per-frame timesteps.
    Returns the latent-space v-prediction [B,T,H,W,C].
    """
    dit = model.video.dit
    xv = model.video.patchify(x_lat)
    time_cond = model.video.get_time_cond(t)
    for L in range(model.num_layers):
        xv = dit.blocks[L](xv, time_cond, num_views=dit.num_views)
        xv = model.adapters[L].update_video(xv, ctx, ctx_mask)
    return model.video.head(xv, time_cond)


@torch.no_grad()
def rollout_future_latents(model: STWAMModel, z_hist: torch.Tensor,
                           ctx: torch.Tensor | None, ctx_mask: torch.Tensor | None,
                           num_steps: int | None = None) -> torch.Tensor:
    """Diffusion-forcing DDIM rollout of the future frames.

    z_hist: [B, num_history, H, W, C] clean history latents.
    Returns the full clean window [B, n_frames, H, W, C] in fp32.

    All future frames share one noise level per step (training sampled t
    i.i.d. per frame, so any combination — including all-equal — is
    in-distribution).  The sampling schedule is the plain linear index grid of
    ``ddim_sample_step`` (no time_dist_shift: that only shapes the *training*
    t distribution).
    """
    diff = model.diffusion
    num_steps = num_steps or diff.sampling_timesteps
    B, nh = z_hist.shape[0], z_hist.shape[1]
    nf = model.config.n_frames - nh
    if nf <= 0:
        return z_hist.float()
    device = z_hist.device
    z_hist = z_hist.float()

    # [-1, ..., timesteps-1]; index 0 is the "clean" endpoint.
    steps = torch.linspace(-1, diff.timesteps - 1, num_steps + 1,
                           device=device, dtype=torch.long)
    x_fut = torch.randn(B, nf, *z_hist.shape[2:], device=device)
    t_hist = torch.zeros(B, nh, dtype=torch.long, device=device)
    one = torch.ones((), device=device)

    for i in range(num_steps, 0, -1):
        t_cur, t_next = steps[i], steps[i - 1]
        t_in = torch.cat(
            [t_hist, t_cur.expand(B, nf)], dim=1)                  # [B, nh+nf]
        x_in = torch.cat([z_hist, x_fut], dim=1)
        v_fut = video_only_forward(model, x_in, t_in, ctx, ctx_mask)[:, nh:].float()

        ac = diff.alphas_cumprod[t_cur].float()
        ac_next = diff.alphas_cumprod[t_next].float() if t_next >= 0 else one
        # v-pred -> x0 -> eps -> x_{t_next} (ddim_sample_step algebra)
        x0 = ac.sqrt() * x_fut - (1 - ac).sqrt() * v_fut
        eps = ((1 / ac).sqrt() * x_fut - x0) / ((1 / ac) - 1).sqrt()
        x_fut = ac_next.sqrt() * x0 + (1 - ac_next).sqrt() * eps

    return torch.cat([z_hist, x_fut], dim=1)


@torch.no_grad()
def prefill_video_window(model: STWAMModel, window_lat: torch.Tensor,
                         ctx: torch.Tensor | None, ctx_mask: torch.Tensor | None,
                         num_ctx_frames: int) -> list:
    """``STWAMModel.prefill_video`` over a full clean window, caching K/V from
    only the first ``num_ctx_frames`` frames (the adapters' runtime knob).

    K/V are taken post-block, pre-text-update, matching ``coupled_forward``.
    """
    dit = model.video.dit
    B, T = window_lat.shape[0], window_lat.shape[1]
    t_video = torch.zeros(B, T, dtype=torch.long, device=window_lat.device)
    xv = model.video.patchify(window_lat)
    time_cond = model.video.get_time_cond(t_video)
    cache = []
    for L in range(model.num_layers):
        xv = dit.blocks[L](xv, time_cond, num_views=dit.num_views)
        cache.append(model.adapters[L].video_kv(xv[:, :num_ctx_frames]))
        xv = model.adapters[L].update_video(xv, ctx, ctx_mask)
    return cache


@torch.no_grad()
def sample_actions_rollout(model: STWAMModel, z_hist: torch.Tensor,
                           ctx: torch.Tensor | None = None,
                           ctx_mask: torch.Tensor | None = None,
                           action_ctx_frames: int | None = None,
                           video_steps: int | None = None,
                           action_steps: int | None = None) -> torch.Tensor:
    """Rollout the future, then flow-match the action chunk against the
    (possibly extended) K/V window.  Returns [B, chunk_size, action_dim].

    ``action_ctx_frames=num_history`` reproduces ``sample_actions`` up to
    numerics (temporal attention is causal, so history features are identical
    in the longer window); ``None`` reads the full ``n_frames`` window.
    """
    num_steps = action_steps or model.config.action_sampling_steps
    action_ctx_frames = action_ctx_frames or model.config.n_frames
    B = z_hist.shape[0]
    device = z_hist.device
    # Draw the action init noise before the rollout noise so a fixed torch
    # seed matches sample_actions' RNG stream (its first randn is this one).
    x_a = torch.randn(B, model.config.chunk_size, model.config.action_dim, device=device)

    window = rollout_future_latents(model, z_hist, ctx, ctx_mask, num_steps=video_steps)
    cache = prefill_video_window(model, window, ctx, ctx_mask, action_ctx_frames)

    a_ctx = model.action.embed_context(ctx)
    schedule = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    for i in range(num_steps):
        t_cur, t_next = schedule[i], schedule[i + 1]
        dt = (t_cur - t_next)
        ca = model.action.cond(torch.full((B,), float(t_cur), device=device))
        xa = model.action.embed(x_a)
        for L in range(model.num_layers):
            xa = model.action.blocks[L](xa, ca, a_ctx, ctx_mask)
            xa = model.adapters[L].update_action(xa, *cache[L], model.action_times)
        v_a = model.action.head_out(xa)
        x_a = x_a - dt * v_a
    return x_a
