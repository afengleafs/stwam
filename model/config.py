"""STWAM configuration (plain dataclass; the lerobot policy wraps it).

Fields default to the values verified against semantic-wm's `launch.py`
(objective="ddpm" v-pred, temporal_mode="factored", num_history=2, wide_head,
decoder_dim=2048) and the adapter-96 / V-JEPA recipe.  Architecture fields are
normally overwritten by `checkpoint.introspect(DiT-S_D96.pt)`.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class STWAMConfig:
    # --- video expert (semantic-wm DiT-S, loads DiT-S_D96) ---
    video_dim: int = 384
    in_channels: int = 96            # adapter-96 latent
    patch_size: int = 1
    num_layers: int = 12
    num_heads: int = 6               # head_dim 64
    wide_head: bool = True
    decoder_dim: int = 2048
    temporal_mode: str = "factored"  # branch B
    max_frames: int = 16
    num_views: int = 2
    action_dropout_prob: float = 0.1

    # --- action expert (light, from scratch) ---
    action_dim: int = 7              # robot/policy action dim from LIBERO
    pretrained_action_dim: int = 10  # only for loading semantic-wm DiT action_embedder
    action_hidden: int = 128
    action_layers: int = 12
    action_heads: int = 4

    # --- joint MoT adapter ---
    mot_da: int = 384
    mot_heads: int = 6

    # --- semantic encoder (V-JEPA + frozen S-VAE adapter) ---
    vjepa_model_size: str = "vitl"
    vjepa2_ckpt: str | None = "weights/vjepa/vjepa2_1_vitl_dist_vitG_384.pt"
    vjepa_input_size: int = 256
    semantic_dim: int = 96
    adapter_latent_dim: int = 96
    adapter_num_heads: int = 12
    adapter_num_layers: int = 3
    adapter_intermediate_size: int = 3072
    freeze_vjepa: bool = True
    freeze_adapter: bool = True

    # --- diffusion (must match DiT-S_D96 training) ---
    objective: str = "ddpm"          # v-prediction Diffusion (launch.py default)
    timesteps: int = 1000
    sampling_timesteps: int = 10
    time_dist_shift: float = 2.45    # ~sqrt(24576/4096) for non-VAE latent
    num_history: int = 2

    # --- action diffusion (flow matching over the chunk) ---
    action_sampling_steps: int = 10

    # --- text / state conditioning ---
    use_language: bool = True
    precomputed_text_embeddings: bool = True
    text_encoder_name: str | None = "google/flan-t5-large"
    text_dim: int = 1024
    proprio_dim: int = 8
    max_state_dim: int = 8

    # --- action chunking / time window ---
    chunk_size: int = 16
    n_action_steps: int = 8
    n_frames: int = 8
    frame_skip: int = 1
    observation_delta_indices: tuple[int, ...] | None = None
    action_delta_indices: tuple[int, ...] | None = None

    # --- loss weights ---
    loss_lambda_video: float = 1.0
    loss_lambda_action: float = 1.0

    # --- paths ---
    video_dit_ckpt: str | None = None
    adapter_ckpt: str | None = None

    # --- runtime ---
    device: str = "cuda"
    dtype: str = "bfloat16"

    def apply_introspection(self, info: dict) -> None:
        """Overwrite architecture fields from `checkpoint.introspect` output."""
        pairs = [("dim", "video_dim"), ("in_channels", "in_channels"),
                 ("patch_size", "patch_size"), ("num_layers", "num_layers"),
                 ("num_heads", "num_heads"), ("decoder_dim", "decoder_dim")]
        for k_ck, k_cfg in pairs:
            if info.get(k_ck) is not None:
                setattr(self, k_cfg, info[k_ck])
        if info.get("action_dim") is not None:
            self.pretrained_action_dim = info["action_dim"]
        if info.get("wide_head") is not None:
            self.wide_head = info["wide_head"]
