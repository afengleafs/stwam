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


def _per_frame_action(action: torch.Tensor, T: int) -> torch.Tensor:
    """Map an action chunk [B,S,a] to T per-frame conditioning actions [B,T,a]."""
    B, S, A = action.shape
    if S == T:
        return action
    idx = torch.linspace(0, S - 1, T, device=action.device).round().long()
    return action[:, idx]


class STWAMPolicy(_Base):
    config_class = STWAMConfig
    name = "stwam"

    def __init__(self, config: STWAMConfig, vjepa_encoder: nn.Module | None = None,
                 dataset_stats: dict | None = None) -> None:
        if _HAS_LEROBOT:
            super().__init__(config)
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
    def _encode_video(self, batch: dict) -> torch.Tensor:
        """Return semantic latent z [B,T,16,16,96] from raw frames or precomputed."""
        if "semantic_latent" in batch:
            return batch["semantic_latent"]
        video = batch.get("video", batch.get("observation.images"))
        assert video is not None, "batch needs 'video' [B,3,T,H,W] or 'semantic_latent'"
        assert self.model.vjepa is not None, "no V-JEPA encoder; pass precomputed 'semantic_latent'"
        return self.model.vjepa.encode(video)

    def _context(self, batch: dict):
        ctx = batch.get("text_embeds")          # [B,L,4096] precomputed
        ctx_mask = batch.get("text_mask")
        proprio = batch.get("observation.state", batch.get("proprio"))
        return self.model.build_context(ctx, ctx_mask, proprio)

    # ------------------------------------------------------------------ training
    def forward(self, batch: dict) -> tuple[torch.Tensor, dict]:
        z = self._encode_video(batch)                       # [B,T,H,W,C]
        action = batch["action"]                            # [B,chunk,a]
        video_action = batch.get("video_action", _per_frame_action(action, z.shape[1]))
        ctx, ctx_mask = self._context(batch)
        return self.model.training_loss(
            z, action, video_action, ctx, ctx_mask,
            action_is_pad=batch.get("action_is_pad"),
            image_is_pad=batch.get("image_is_pad"),
        )

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def predict_action_chunk(self, batch: dict) -> torch.Tensor:
        z = self._encode_video(batch)
        anchor = z[:, : self.config.num_history]
        va = torch.zeros(z.shape[0], self.config.num_history, self.config.action_dim, device=z.device)
        ctx, ctx_mask = self._context(batch)
        return self.model.sample_actions(anchor, va, ctx, ctx_mask)

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
