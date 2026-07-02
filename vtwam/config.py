"""VTWAM configuration.

VTWAM is the pixel/VAE-latent control ablation for STWAM.  It keeps the same
WAM/action/MoT control path and swaps the visual latent interface from
V-JEPA/S-VAE 96-d semantic latents to semantic-wm's SD3 VAE 16-d latents.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VTWAMConfig:
    # --- video expert (semantic-wm DiT-S VAE, loads vae/DiT-S_D16.pt) ---
    video_dim: int = 384
    in_channels: int = 16
    patch_size: int = 2
    num_layers: int = 12
    num_heads: int = 6
    wide_head: bool = False
    decoder_dim: int = 2048
    temporal_mode: str = "factored"
    max_frames: int = 16
    num_views: int = 2
    action_dropout_prob: float = 0.1

    # --- action expert (same as STWAM, trained from scratch) ---
    action_dim: int = 7
    pretrained_action_dim: int = 10
    action_hidden: int = 128
    action_layers: int = 12
    action_heads: int = 4

    # --- joint MoT adapter ---
    mot_da: int = 384
    mot_heads: int = 6

    # --- pixel/VAE encoder ---
    vae_model_dir: str | None = "vtwam/checkpoint/sd3-medium-diffusers"
    freeze_vae: bool = True
    vae_sample: bool = True
    vae_chunk_size: int = 64

    # --- diffusion (VAE semantic-wm training uses no SD3-style time shift) ---
    objective: str = "ddpm"
    timesteps: int = 1000
    sampling_timesteps: int = 10
    time_dist_shift: float = 1.0
    num_history: int = 1

    # --- action diffusion (flow matching over the chunk) ---
    action_sampling_steps: int = 10

    # --- text / state conditioning ---
    use_language: bool = True
    precomputed_text_embeddings: bool = True
    text_encoder_name: str | None = "google/flan-t5-large"
    text_dim: int = 1024
    proprio_dim: int = 8
    max_state_dim: int = 8

    # --- action chunking / time window (FastWAM LIBERO defaults) ---
    chunk_size: int = 32
    n_action_steps: int = 32
    n_frames: int = 9
    frame_skip: int = 1
    observation_delta_indices: tuple[int, ...] | None = None
    action_delta_indices: tuple[int, ...] | None = None

    # --- loss weights ---
    loss_lambda_video: float = 1.0
    loss_lambda_action: float = 1.0

    # --- paths ---
    video_dit_ckpt: str | None = "vtwam/checkpoint/vae/DiT-S_D16.pt"

    # --- runtime ---
    device: str = "cuda"
    dtype: str = "bfloat16"

    def apply_introspection(self, info: dict) -> None:
        """Overwrite architecture fields from a semantic-wm DiT state dict."""
        pairs = [
            ("dim", "video_dim"),
            ("in_channels", "in_channels"),
            ("patch_size", "patch_size"),
            ("num_layers", "num_layers"),
            ("num_heads", "num_heads"),
            ("decoder_dim", "decoder_dim"),
        ]
        for k_ck, k_cfg in pairs:
            if info.get(k_ck) is not None:
                setattr(self, k_cfg, info[k_ck])
        if info.get("action_dim") is not None:
            self.pretrained_action_dim = info["action_dim"]
        if info.get("wide_head") is not None:
            self.wide_head = info["wide_head"]
