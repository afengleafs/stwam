"""VTWAM inner model.

The control backbone is intentionally the same as ``STWAMModel``.  This class
only gives the ablation a distinct type/name and attaches the VAE encoder under
``model.vae`` for the policy wrapper.
"""
from __future__ import annotations

import torch.nn as nn

from model.modeling_stwam import STWAMModel

from .config import VTWAMConfig


class VTWAMModel(STWAMModel):
    def __init__(self, config: VTWAMConfig, vae_encoder: nn.Module | None = None) -> None:
        super().__init__(config, vjepa_encoder=None)
        self.vae = vae_encoder
