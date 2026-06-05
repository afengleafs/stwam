"""STWAM inner model: couples the (pretrained, frozen-structure) video DiT with
the light action expert through per-layer zero-init joint MoT adapters, and
implements training loss + action sampling.

Latent convention (semantic-wm): video latents are `[B, T, H, W, C]`.

Video denoising follows the vendored v-prediction `Diffusion` (diffusion-forcing:
independent per-frame noise during training; clean history is an *inference*
concept).  Action denoising is a small flow-matching process over the chunk.
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
            num_layers=config.num_layers, num_heads=config.num_heads, action_dim=config.action_dim,
            max_frames=config.max_frames, wide_head=config.wide_head, decoder_dim=config.decoder_dim,
            temporal_mode=config.temporal_mode, action_dropout_prob=config.action_dropout_prob,
        )
        self.action = ActionExpert(
            action_dim=config.action_dim, dim=config.action_hidden, num_layers=config.action_layers,
            num_heads=config.action_heads, ctx_dim=config.text_dim,
        )
        self.adapters = nn.ModuleList([
            JointMoTAdapter(
                video_dim=config.video_dim, action_dim=config.action_hidden, da=config.mot_da,
                num_heads=config.mot_heads, ctx_dim=config.text_dim,
                video_reads_action=config.video_reads_action,
            ) for _ in range(config.num_layers)
        ])
        assert self.video.num_layers == self.action.num_layers == len(self.adapters)
        self.num_layers = config.num_layers
        self.diffusion = Diffusion(
            timesteps=config.timesteps, sampling_timesteps=config.sampling_timesteps,
            time_dist_shift=config.time_dist_shift,
        )
        self.vjepa = vjepa_encoder
        self.proprio_encoder = (
            nn.Linear(config.proprio_dim, config.text_dim) if config.proprio_dim > 0 else None)

    # ------------------------------------------------------------------ coupled forward
    def coupled_forward(self, xv_lat: torch.Tensor, t: torch.Tensor, video_action: torch.Tensor,
                        a_tokens_in: torch.Tensor, t_action: torch.Tensor,
                        ctx: torch.Tensor | None, ctx_mask: torch.Tensor | None):
        """Interleave video DiT blocks, action blocks, and joint adapters.

        Returns (v_video [B,T,H,W,C], v_action [B,chunk,action_dim]).
        """
        dit = self.video.dit
        xv = self.video.patchify(xv_lat)                 # [B,T,h,w,dim]
        time_cond, action_cond = self.video.get_cond(t, video_action)
        c = time_cond + action_cond
        xa = self.action.embed(a_tokens_in)              # [B,chunk,da]
        ca = self.action.cond(t_action)                  # [B,da]
        a_ctx = self.action.embed_context(ctx)           # [B,L,da] | None
        for L in range(self.num_layers):
            xv = dit.blocks[L](xv, c, num_views=dit.num_views)
            xa = self.action.blocks[L](xa, ca, a_ctx, ctx_mask)
            xv, xa = self.adapters[L](xv, xa, ctx, ctx_mask)
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
                      video_action: torch.Tensor, ctx: torch.Tensor | None = None,
                      ctx_mask: torch.Tensor | None = None, action_is_pad: torch.Tensor | None = None,
                      image_is_pad: torch.Tensor | None = None):
        B, T, H, W, C = z.shape
        # --- video: v-prediction diffusion-forcing (per-frame independent t) ---
        t = self.diffusion.sample_t(B, T, device=z.device)        # [B,T] long
        noise = torch.randn_like(z)
        x_video_t = self.diffusion.q_sample(z, t, noise)
        ac = self.diffusion.alphas_cumprod[t.reshape(-1)].view(B, T, 1, 1, 1)
        target_v = ac.sqrt() * noise - (1 - ac).sqrt() * z

        # --- action: flow matching over the chunk (per-sample t) ---
        Bc, S, A = action_chunk.shape
        t_a = torch.rand(Bc, device=z.device)
        noise_a = torch.randn_like(action_chunk)
        x_a_t = torch.lerp(action_chunk, noise_a, t_a.view(Bc, 1, 1))
        target_a = noise_a - action_chunk

        v_video, v_action = self.coupled_forward(x_video_t, t, video_action, x_a_t, t_a, ctx, ctx_mask)

        # video loss (optional per-frame pad mask)
        v_err = (v_video - target_v).pow(2).mean(dim=(2, 3, 4))     # [B,T]
        if image_is_pad is not None:
            keep = (~image_is_pad).float()
            loss_video = (v_err * keep).sum() / keep.sum().clamp(min=1)
        else:
            loss_video = v_err.mean()

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
    def sample_actions(self, anchor_lat: torch.Tensor, video_action: torch.Tensor,
                       ctx: torch.Tensor | None = None, ctx_mask: torch.Tensor | None = None,
                       num_steps: int | None = None) -> torch.Tensor:
        """Flow-matching action denoising conditioned on the clean anchor latent.

        anchor_lat: [B, T_ctx, H, W, C] (clean world-model context, t=0).
        Returns action chunk [B, chunk_size, action_dim].
        """
        num_steps = num_steps or self.config.action_sampling_steps
        B, Tc = anchor_lat.shape[0], anchor_lat.shape[1]
        device = anchor_lat.device
        x_a = torch.randn(B, self.config.chunk_size, self.config.action_dim, device=device)
        t_video = torch.zeros(B, Tc, dtype=torch.long, device=device)
        schedule = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
        for i in range(num_steps):
            t_cur, t_next = schedule[i], schedule[i + 1]
            dt = (t_cur - t_next)
            t_a = torch.full((B,), float(t_cur), device=device)
            _, v_a = self.coupled_forward(anchor_lat, t_video, video_action, x_a, t_a, ctx, ctx_mask)
            x_a = x_a - dt * v_a
        return x_a
