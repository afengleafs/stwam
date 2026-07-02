"""Pixel/VAE encoder for VTWAM.

Uses the same SD3 VAE path as semantic-wm's ``encoder_type=vae`` and returns
latents in the repository's canonical video layout: ``[B, T, H, W, C]``.
"""
from __future__ import annotations

from pathlib import Path

import einops
import torch
import torch.nn as nn


def _dtype(name: str) -> torch.dtype:
    table = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if name not in table:
        raise ValueError(f"unknown dtype {name!r}")
    return table[name]


def _resolve_pretrained_args(model_dir: str | Path) -> tuple[str, str | None]:
    path = Path(model_dir)
    if (path / "vae" / "config.json").is_file():
        return str(path), "vae"
    if (path / "config.json").is_file():
        return str(path), None
    raise FileNotFoundError(
        f"Could not find SD3 VAE config under {path}. Expected either "
        f"{path / 'vae' / 'config.json'} or {path / 'config.json'}."
    )


class VAEVideoEncoder(nn.Module):
    temporal_downsample_factor = 1

    def __init__(self, config) -> None:
        super().__init__()
        if not config.vae_model_dir:
            raise ValueError("config.vae_model_dir must point to the local SD3 VAE directory")

        from diffusers.models import AutoencoderKL

        pretrained, subfolder = _resolve_pretrained_args(config.vae_model_dir)
        torch_dtype = _dtype(getattr(config, "dtype", "bfloat16"))
        kwargs = {"local_files_only": True, "torch_dtype": torch_dtype}
        if subfolder is not None:
            kwargs["subfolder"] = subfolder
        self.vae = AutoencoderKL.from_pretrained(pretrained, **kwargs)
        self.freeze_vae = bool(getattr(config, "freeze_vae", True))
        self.vae.eval()
        if self.freeze_vae:
            self.vae.requires_grad_(False)

        self.sample = bool(getattr(config, "vae_sample", True))
        self.chunk_size = int(getattr(config, "vae_chunk_size", 64))
        self.z_dim = int(self.vae.config.latent_channels)
        self.scaling_factor = float(self.vae.config.scaling_factor)

    def train(self, mode: bool = True) -> "VAEVideoEncoder":
        super().train(mode)
        if self.freeze_vae:
            self.vae.eval()
        return self

    def _chunked(self, fn, x: torch.Tensor) -> torch.Tensor:
        chunk = max(self.chunk_size, 1)
        if x.shape[0] <= chunk:
            return fn(x)
        return torch.cat([fn(x[i:i + chunk]) for i in range(0, x.shape[0], chunk)])

    @staticmethod
    def _param_device_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
        p = next(module.parameters())
        return p.device, p.dtype

    @torch.no_grad()
    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """video: ``[B, 3, T, H, W]`` in ``[0,1]`` -> ``[B,T,H/8,W/8,16]``."""
        if video.dtype == torch.uint8:
            video = video.float().div_(255.0)
        else:
            video = video.float()
            if video.numel() > 0 and video.detach().amax() > 2:
                video = video / 255.0

        B, C, T, H, W = video.shape
        if C != 3:
            raise ValueError(f"expected RGB video [B,3,T,H,W], got {tuple(video.shape)}")
        x_in = einops.rearrange(video, "b c t h w -> (b t) c h w")
        x_in = x_in * 2 - 1
        dev, dtype = self._param_device_dtype(self.vae)
        x_in = x_in.to(device=dev, dtype=dtype)

        def _encode(x):
            dist = self.vae.encode(x).latent_dist
            return dist.sample() if self.sample else dist.mode()

        z = self._chunked(_encode, x_in) * self.scaling_factor
        z = einops.rearrange(z, "(b t) c h w -> b t h w c", b=B, t=T)
        return z.float()

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode ``[B,T,h,w,16]`` latents back to RGB ``[B,T,H,W,3]``."""
        B, T, H, W, C = z.shape
        z_in = einops.rearrange(z, "b t h w c -> (b t) c h w")
        dev, dtype = self._param_device_dtype(self.vae)
        z_in = z_in.to(device=dev, dtype=dtype) / self.scaling_factor

        x = self._chunked(lambda y: self.vae.decode(y, return_dict=False)[0], z_in)
        x = (x + 1) / 2
        x = einops.rearrange(x, "(b t) c h w -> b t h w c", b=B, t=T)
        return x.float().clamp(0, 1)
