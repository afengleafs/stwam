"""STWAM policy — lerobot pi05-style wrapper.

Mirrors the `PI05Policy` organization: a `PreTrainedPolicy`-like outer class with
`forward(batch) -> (loss, dict)`, `select_action`/`predict_action_chunk` driven
by an action queue, `reset`, and `get_optim_params`.  `lerobot` is imported
softly so the module is usable standalone (falls back to `nn.Module`).
"""
from __future__ import annotations

from collections import deque

import torch
import torch.nn as nn

from model.config import STWAMConfig
from model.modeling_stwam import STWAMModel

try:  # optional lerobot base
    from lerobot.policies.pretrained import PreTrainedPolicy as _Base
    _HAS_LEROBOT = True
except Exception:  # pragma: no cover
    _Base = nn.Module
    _HAS_LEROBOT = False


class STWAMPolicy(_Base):
    config_class = STWAMConfig
    name = "stwam"

    def __init__(self, config: STWAMConfig, vjepa_encoder: nn.Module | None = None,
                 dataset_stats: dict | None = None) -> None:
        if _HAS_LEROBOT:
            try:
                super().__init__(config)
            except Exception:
                # Keep this wrapper usable with standalone training scripts even
                # when lerobot is installed but expects a native PolicyConfig.
                nn.Module.__init__(self)
        else:
            nn.Module.__init__(self)
        self.config = config
        if vjepa_encoder is None and config.adapter_ckpt:
            from model.vjepa_encoder import VJEPASemanticEncoder
            vjepa_encoder = VJEPASemanticEncoder(config)
        self.model = STWAMModel(config, vjepa_encoder=vjepa_encoder)
        # normalization (lerobot Normalize/Unnormalize) is wired by make_*_processors;
        # kept external so this class stays runnable without lerobot.
        self.reset()

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _canonical_video(video: torch.Tensor) -> torch.Tensor:
        """Normalize common video layouts to [B, 3, T, H, W] in [0, 1]."""
        if video.dtype == torch.uint8:
            video = video.float().div_(255.0)
        else:
            video = video.float()
            if video.numel() > 0 and video.detach().amax() > 2:
                video = video / 255.0
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError(f"expected 5D video tensor, got shape {tuple(video.shape)}")
        if video.shape[1] == 3:      # [B,3,T,H,W]
            return video.contiguous()
        if video.shape[2] == 3:      # [B,T,3,H,W]
            return video.permute(0, 2, 1, 3, 4).contiguous()
        if video.shape[-1] == 3:     # [B,T,H,W,3]
            return video.permute(0, 4, 1, 2, 3).contiguous()
        raise ValueError(f"cannot infer channel axis for video shape {tuple(video.shape)}")

    def _encode_video(self, batch: dict) -> torch.Tensor:
        """Return semantic latent z [B,T,16,16*num_views,96] from video inputs."""
        if "semantic_latent" in batch:
            return batch["semantic_latent"]
        assert self.model.vjepa is not None, "no V-JEPA encoder; pass precomputed 'semantic_latent'"
        if self.config.num_views == 2 and "observation.images.image2" in batch:
            video1 = self._canonical_video(batch["observation.images.image"])
            video2 = self._canonical_video(batch["observation.images.image2"])
            z1 = self.model.vjepa.encode(video1)
            z2 = self.model.vjepa.encode(video2)
            return torch.cat([z1, z2], dim=3)
        video = batch.get("video", batch.get("observation.images", batch.get("observation.images.image")))
        assert video is not None, "batch needs video inputs or 'semantic_latent'"
        return self.model.vjepa.encode(self._canonical_video(video))

    def _context(self, batch: dict):
        ctx = batch.get("text_embeds")          # [B,L,text_dim] precomputed
        ctx_mask = batch.get("text_mask")
        proprio = batch.get("observation.state", batch.get("proprio"))
        return self.model.build_context(ctx, ctx_mask, proprio)

    # ------------------------------------------------------------------ training
    def forward(self, batch: dict) -> tuple[torch.Tensor, dict]:
        z = self._encode_video(batch)                       # [B,T,H,W,C]
        action = batch["action"]                            # [B,chunk,a]
        ctx, ctx_mask = self._context(batch)
        # Pooled connector modes use the two-phase ablation loss (mathematically
        # identical to interleaved coupled_forward when k_draws=1 and no dropout).
        if getattr(self.config, "pooled_adaln", "off") != "off":
            from model.ablation import training_loss_ablation
            return training_loss_ablation(
                self.model, z, action, ctx, ctx_mask,
                k_draws=1, proprio_dropout=0.0,
                action_is_pad=batch.get("action_is_pad"),
                image_is_pad=batch.get("image_is_pad"),
            )
        return self.model.training_loss(
            z, action, ctx, ctx_mask,
            action_is_pad=batch.get("action_is_pad"),
            image_is_pad=batch.get("image_is_pad"),
        )

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def predict_action_chunk(self, batch: dict) -> torch.Tensor:
        z = self._encode_video(batch)
        anchor = z[:, : self.config.num_history]
        ctx, ctx_mask = self._context(batch)
        return self.model.sample_actions(anchor, ctx, ctx_mask)

    @torch.no_grad()
    def select_action(self, batch: dict) -> torch.Tensor:
        if len(self._action_queue) == 0:
            chunk = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(chunk.transpose(0, 1))  # [n_steps, B, a]
        return self._action_queue.popleft()

    def reset(self) -> None:
        self._action_queue: deque = deque(maxlen=self.config.n_action_steps)

    def get_optim_params(self):
        return self.parameters()
