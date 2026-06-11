"""STWAM inner model: couples the (pretrained, frozen-structure) video DiT with
the light action expert through per-layer zero-init joint MoT adapters, and
implements training loss + action sampling.

Latent convention (semantic-wm): video latents are `[B, T, H, W, C]`.

Video denoising follows the vendored v-prediction `Diffusion` with
diffusion-forcing on the *future* frames; the first `num_history` frames are
always teacher-forced clean (t=0, exact latents) so that what the action expert
reads at training time is exactly what it sees at inference.  The video tower
is not directly action-conditioned (FastWAM-style); language/proprio state
enter through context cross-attention.  Action denoising is a small
flow-matching process over the chunk; sampling prefills the video tower once
and reuses the cached per-layer K/V across all flow steps.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._swm.training.diffusion import Diffusion
from .action_expert import ActionExpert
from .config import STWAMConfig
from .mot_adapter import JointMoTAdapter
from .video_expert import VideoExpert


class STWAMModel(nn.Module):
    def __init__(self, config: STWAMConfig, vjepa_encoder: nn.Module | None = None) -> None:
        super().__init__()
        self.config = config
        self.video = VideoExpert(
            in_channels=config.in_channels, patch_size=config.patch_size, dim=config.video_dim,
            num_layers=config.num_layers, num_heads=config.num_heads, action_dim=config.pretrained_action_dim,
            max_frames=config.max_frames, wide_head=config.wide_head, decoder_dim=config.decoder_dim,
            num_views=config.num_views, temporal_mode=config.temporal_mode,
            action_dropout_prob=config.action_dropout_prob,
        )
        self.action = ActionExpert(
            action_dim=config.action_dim, dim=config.action_hidden, num_layers=config.action_layers,
            num_heads=config.action_heads, ctx_dim=config.text_dim,
        )
        self.adapters = nn.ModuleList([
            JointMoTAdapter(
                video_dim=config.video_dim, action_dim=config.action_hidden, da=config.mot_da,
                num_heads=config.mot_heads, ctx_dim=config.text_dim,
            ) for _ in range(config.num_layers)
        ])
        assert self.video.num_layers == self.action.num_layers == len(self.adapters)
        self.num_layers = config.num_layers
        # Action-chunk positions mapped onto the video frame-time axis (for the
        # adapter's time-aligned RoPE): step j sits between the last observed
        # frame (num_history-1) and the end of the predicted window (n_frames-1).
        span = max(config.n_frames - config.num_history, 1)
        times = (config.num_history - 1) + (torch.arange(config.chunk_size, dtype=torch.float32) + 1) * span / config.chunk_size
        self.register_buffer("action_times", times, persistent=False)
        self.diffusion = Diffusion(
            timesteps=config.timesteps, sampling_timesteps=config.sampling_timesteps,
            time_dist_shift=config.time_dist_shift,
        )
        self.vjepa = vjepa_encoder
        self.proprio_encoder = (
            nn.Linear(config.proprio_dim, config.text_dim) if config.proprio_dim > 0 else None)

    # ------------------------------------------------------------------ coupled forward
    def coupled_forward(self, xv_lat: torch.Tensor, t: torch.Tensor,
                        a_tokens_in: torch.Tensor, t_action: torch.Tensor,
                        ctx: torch.Tensor | None, ctx_mask: torch.Tensor | None):
        """Interleave video DiT blocks, action blocks, and joint adapters.

        The action expert reads only the first `num_history` (clean context)
        frames through the adapters; the video tower is not directly
        action-conditioned and reads language/state through the adapters.
        Returns (v_video [B,T,H,W,C], v_action [B,chunk,action_dim]).
        """
        dit = self.video.dit
        xv = self.video.patchify(xv_lat)                 # [B,T,h,w,dim]
        time_cond = self.video.get_time_cond(t)
        c = time_cond
        xa = self.action.embed(a_tokens_in)              # [B,chunk,da]
        ca = self.action.cond(t_action)                  # [B,da]
        a_ctx = self.action.embed_context(ctx)           # [B,L,da] | None
        num_ctx = min(self.config.num_history, xv_lat.shape[1])
        for L in range(self.num_layers):
            xv = dit.blocks[L](xv, c, num_views=dit.num_views)
            xa = self.action.blocks[L](xa, ca, a_ctx, ctx_mask)
            xv, xa = self.adapters[L](xv, xa, ctx, ctx_mask, num_ctx, self.action_times)
        v_video = self.video.head(xv, time_cond)
        v_action = self.action.head_out(xa)
        return v_video, v_action

    # ------------------------------------------------------------------ context
    def build_context(self, text_ctx: torch.Tensor | None, text_mask: torch.Tensor | None,
                      proprio: torch.Tensor | None):
        """Append a proprio/state token to the (precomputed) text context."""
        if proprio is not None and self.proprio_encoder is not None:
            tok = self.proprio_encoder(proprio).unsqueeze(1)  # [B,1,text_dim]
            if text_ctx is None:
                return tok, torch.ones(tok.shape[0], 1, dtype=torch.bool, device=tok.device)
            text_ctx = torch.cat([text_ctx, tok], dim=1)
            m = torch.ones(tok.shape[0], 1, dtype=torch.bool, device=tok.device)
            text_mask = m if text_mask is None else torch.cat([text_mask, m], dim=1)
        return text_ctx, text_mask

    # ------------------------------------------------------------------ training loss
    def training_loss(self, z: torch.Tensor, action_chunk: torch.Tensor,
                      ctx: torch.Tensor | None = None,
                      ctx_mask: torch.Tensor | None = None, action_is_pad: torch.Tensor | None = None,
                      image_is_pad: torch.Tensor | None = None):
        B, T, H, W, C = z.shape
        nh = min(self.config.num_history, T)
        # --- video: v-prediction diffusion-forcing on the FUTURE frames; the
        # history frames are teacher-forced clean (t=0, exact latents) so the
        # context the action expert reads matches inference exactly ---
        t = self.diffusion.sample_t(B, T, device=z.device)        # [B,T] long
        t[:, :nh] = 0
        noise = torch.randn_like(z)
        x_video_t = self.diffusion.q_sample(z, t, noise)
        x_video_t[:, :nh] = z[:, :nh]
        ac = self.diffusion.alphas_cumprod[t.reshape(-1)].view(B, T, 1, 1, 1)
        target_v = ac.sqrt() * noise - (1 - ac).sqrt() * z

        # --- action: flow matching over the chunk (per-sample t) ---
        Bc, S, A = action_chunk.shape
        t_a = torch.rand(Bc, device=z.device)
        noise_a = torch.randn_like(action_chunk)
        x_a_t = torch.lerp(action_chunk, noise_a, t_a.view(Bc, 1, 1))
        target_a = noise_a - action_chunk

        v_video, v_action = self.coupled_forward(x_video_t, t, x_a_t, t_a, ctx, ctx_mask)

        # video loss on future frames only (history is teacher-forced clean)
        v_err = (v_video - target_v).pow(2).mean(dim=(2, 3, 4))     # [B,T]
        keep = torch.ones(B, T, device=z.device)
        keep[:, :nh] = 0
        if image_is_pad is not None:
            keep = keep * (~image_is_pad).float()
        loss_video = (v_err * keep).sum() / keep.sum().clamp(min=1)

        # action loss (optional per-step pad mask)
        a_err = (v_action - target_a).pow(2).mean(dim=2)            # [B,S]
        if action_is_pad is not None:
            keep = (~action_is_pad).float()
            loss_action = (a_err * keep).sum() / keep.sum().clamp(min=1)
        else:
            loss_action = a_err.mean()

        total = self.config.loss_lambda_video * loss_video + self.config.loss_lambda_action * loss_action
        return total, {"loss_video": float(loss_video.detach()),
                       "loss_action": float(loss_action.detach())}

    # ------------------------------------------------------------------ action sampling
    @torch.no_grad()
    def prefill_video(self, anchor_lat: torch.Tensor, ctx: torch.Tensor | None,
                      ctx_mask: torch.Tensor | None) -> list:
        """Run the (action-independent) video tower once over the clean context.

        Returns the per-layer adapter K/V cache for the action read path.
        Mirrors `coupled_forward`'s video side: K/V are taken from the post-block
        features *before* the gated text update.
        """
        dit = self.video.dit
        B, Tc = anchor_lat.shape[0], anchor_lat.shape[1]
        t_video = torch.zeros(B, Tc, dtype=torch.long, device=anchor_lat.device)
        xv = self.video.patchify(anchor_lat)
        time_cond = self.video.get_time_cond(t_video)
        c = time_cond
        cache = []
        for L in range(self.num_layers):
            xv = dit.blocks[L](xv, c, num_views=dit.num_views)
            cache.append(self.adapters[L].video_kv(xv))
            xv = self.adapters[L].update_video(xv, ctx, ctx_mask)
        return cache

    @torch.no_grad()
    def sample_actions(self, anchor_lat: torch.Tensor, ctx: torch.Tensor | None = None,
                       ctx_mask: torch.Tensor | None = None,
                       num_steps: int | None = None) -> torch.Tensor:
        """Flow-matching action denoising conditioned on the clean anchor latent.

        anchor_lat: [B, T_ctx, H, W, C] (clean world-model context, t=0).
        The video tower runs exactly once (prefill); every flow step then only
        runs the light action expert against the cached per-layer K/V.
        Returns action chunk [B, chunk_size, action_dim].
        """
        num_steps = num_steps or self.config.action_sampling_steps
        B = anchor_lat.shape[0]
        device = anchor_lat.device
        cache = self.prefill_video(anchor_lat, ctx, ctx_mask)
        a_ctx = self.action.embed_context(ctx)
        x_a = torch.randn(B, self.config.chunk_size, self.config.action_dim, device=device)
        schedule = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
        for i in range(num_steps):
            t_cur, t_next = schedule[i], schedule[i + 1]
            dt = (t_cur - t_next)
            ca = self.action.cond(torch.full((B,), float(t_cur), device=device))
            xa = self.action.embed(x_a)
            for L in range(self.num_layers):
                xa = self.action.blocks[L](xa, ca, a_ctx, ctx_mask)
                xa = self.adapters[L].update_action(xa, *cache[L], self.action_times)
            v_a = self.action.head_out(xa)
            x_a = x_a - dt * v_a
        return x_a
