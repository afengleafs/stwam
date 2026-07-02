"""VTWAM policy wrapper for the pixel/VAE latent ablation."""
from __future__ import annotations

from collections import deque

import torch
import torch.nn as nn

from .config import VTWAMConfig
from .modeling_vtwam import VTWAMModel

try:
    from lerobot.policies.pretrained import PreTrainedPolicy as _Base
    _HAS_LEROBOT = True
except Exception:  # pragma: no cover
    _Base = nn.Module
    _HAS_LEROBOT = False


class VTWAMPolicy(_Base):
    config_class = VTWAMConfig
    name = "vtwam"

    def __init__(self, config: VTWAMConfig, vae_encoder: nn.Module | None = None,
                 dataset_stats: dict | None = None) -> None:
        if _HAS_LEROBOT:
            try:
                super().__init__(config)
            except Exception:
                nn.Module.__init__(self)
        else:
            nn.Module.__init__(self)
        self.config = config
        if vae_encoder is None and config.vae_model_dir:
            from .vae_encoder import VAEVideoEncoder
            vae_encoder = VAEVideoEncoder(config)
        self.model = VTWAMModel(config, vae_encoder=vae_encoder)
        self.reset()

    @staticmethod
    def _canonical_video(video: torch.Tensor) -> torch.Tensor:
        """Normalize common layouts to ``[B, 3, T, H, W]`` in ``[0, 1]``."""
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
        if video.shape[1] == 3:
            return video.contiguous()
        if video.shape[2] == 3:
            return video.permute(0, 2, 1, 3, 4).contiguous()
        if video.shape[-1] == 3:
            return video.permute(0, 4, 1, 2, 3).contiguous()
        raise ValueError(f"cannot infer channel axis for video shape {tuple(video.shape)}")

    def _encode_video(self, batch: dict) -> torch.Tensor:
        """Return VAE latent z ``[B,T,32,32*num_views,16]`` for 256px inputs."""
        for key in ("vae_latent", "pixel_latent", "latent"):
            if key in batch:
                return batch[key]
        assert self.model.vae is not None, "no VAE encoder; pass precomputed 'vae_latent'"
        if self.config.num_views == 2 and "observation.images.image2" in batch:
            video1 = self._canonical_video(batch["observation.images.image"])
            video2 = self._canonical_video(batch["observation.images.image2"])
            z1 = self.model.vae.encode(video1)
            z2 = self.model.vae.encode(video2)
            return torch.cat([z1, z2], dim=3)
        video = batch.get("video", batch.get("observation.images", batch.get("observation.images.image")))
        assert video is not None, "batch needs video inputs or 'vae_latent'"
        return self.model.vae.encode(self._canonical_video(video))

    def _context(self, batch: dict):
        ctx = batch.get("text_embeds")
        ctx_mask = batch.get("text_mask")
        proprio = batch.get("observation.state", batch.get("proprio"))
        return self.model.build_context(ctx, ctx_mask, proprio)

    def forward(self, batch: dict) -> tuple[torch.Tensor, dict]:
        z = self._encode_video(batch)
        action = batch["action"]
        ctx, ctx_mask = self._context(batch)
        return self.model.training_loss(
            z, action, ctx, ctx_mask,
            action_is_pad=batch.get("action_is_pad"),
            image_is_pad=batch.get("image_is_pad"),
        )

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
            self._action_queue.extend(chunk.transpose(0, 1))
        return self._action_queue.popleft()

    def reset(self) -> None:
        self._action_queue: deque = deque(maxlen=self.config.n_action_steps)

    def get_optim_params(self):
        return self.parameters()
