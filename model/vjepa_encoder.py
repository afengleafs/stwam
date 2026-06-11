"""Semantic encoder: frozen V-JEPA 2.1 backbone + frozen S-VAE adapter (96-dim).

Runs only on the server (needs the local V-JEPA 2.1 checkpoint and the HF
adapter checkpoint `adapter_vjepa_image_96.pt`).  Output latent layout is the
semantic-wm canonical `[B, T, 16, 16, 96]` per camera view.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def _unwrap(sd):
    for k in ("model", "state_dict", "adapter", "ema", "module"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]
    return sd


def _strip_prefixes(sd):
    for pfx in ("module.", "model.", "_orig_mod."):
        if isinstance(sd, dict) and any(k.startswith(pfx) for k in sd):
            sd = {
                k[len(pfx):] if k.startswith(pfx) else k: v
                for k, v in sd.items()
            }
    return sd


class VJEPASemanticEncoder(nn.Module):
    temporal_downsample_factor = 1

    def __init__(self, config) -> None:
        super().__init__()
        # imported lazily: pulls the vendored vjepa2 backbone + ImageNet stats
        from ._swm.models.encoders.vjepa2 import VJEPA2EncoderWrapper
        from ._swm.models.adapters import create_adapter

        if not config.vjepa2_ckpt:
            raise ValueError("config.vjepa2_ckpt must point to the local V-JEPA 2.1 checkpoint")
        vjepa2_ckpt = Path(config.vjepa2_ckpt)
        if not vjepa2_ckpt.is_file():
            raise FileNotFoundError(
                f"V-JEPA 2.1 checkpoint not found: {vjepa2_ckpt}. "
                "Download vjepa2_1_vitl_dist_vitG_384.pt into weights/vjepa first."
            )

        self.backbone = VJEPA2EncoderWrapper(
            model_size=config.vjepa_model_size, checkpoint_path=str(vjepa2_ckpt),
            input_size=config.vjepa_input_size,
        )
        embed_dim = self.backbone.latent_dim  # 1024 for ViT-L
        adapter_ckpt = None
        if config.adapter_ckpt:
            adapter_ckpt = torch.load(
                config.adapter_ckpt, map_location="cpu", weights_only=False
            )
        adapter_cfg = {
            "adapter_type": "svae",
            "adapter_latent_dim": config.adapter_latent_dim,
            "adapter_num_heads": config.adapter_num_heads,
            "adapter_num_layers": config.adapter_num_layers,
            "adapter_intermediate_size": config.adapter_intermediate_size,
        }
        if isinstance(adapter_ckpt, dict) and isinstance(
            adapter_ckpt.get("adapter_config"), dict
        ):
            adapter_cfg.update(adapter_ckpt["adapter_config"])
        self.adapter = create_adapter(adapter_cfg, input_dim=embed_dim)
        if config.adapter_ckpt:
            sd = _strip_prefixes(_unwrap(adapter_ckpt))
            missing, unexpected = self.adapter.load_state_dict(sd, strict=False)
            if missing or unexpected:
                print(f"[VJEPASemanticEncoder] adapter load: "
                      f"{len(missing)} missing / {len(unexpected)} unexpected keys")
        self.z_dim = adapter_cfg["adapter_latent_dim"]
        if config.freeze_vjepa:
            self.backbone.requires_grad_(False)
        if config.freeze_adapter:
            self.adapter.requires_grad_(False)
        self.adapter.eval()

    @torch.no_grad()
    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """video: [B, 3, T, H, W] in [0,1]  ->  z_l: [B, T, 16, 16, 96]."""
        x = video.permute(0, 2, 3, 4, 1).contiguous()   # [B,T,H,W,3]
        z = self.backbone.encode(x)                       # [B,T,16,16,1024]
        self.adapter.eval()
        z_l = self.adapter.encode(z)                      # eval -> latent only
        if isinstance(z_l, (tuple, list)):
            z_l = z_l[0]
        return z_l.float()

    @torch.no_grad()
    def decode_features(self, z_l: torch.Tensor) -> torch.Tensor:
        """Decode 96-d latent back to V-JEPA feature space (pixel decode needs the
        separate pixel_decoder, loaded from the same adapter checkpoint family)."""
        return self.adapter.decode(z_l)
