"""Full VTWAM verification; needs local VAE DiT + SD3 VAE weights."""
from __future__ import annotations

import sys

import torch

from model import checkpoint as ck

from .config import VTWAMConfig
from .modeling_vtwam import VTWAMModel
from .vae_encoder import VAEVideoEncoder


def main() -> None:
    dit_ckpt = sys.argv[1] if len(sys.argv) > 1 else "vtwam/checkpoint/vae/DiT-S_D16.pt"
    vae_dir = sys.argv[2] if len(sys.argv) > 2 else "vtwam/checkpoint/sd3-medium-diffusers"
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    sd = ck.load_raw_state_dict(dit_ckpt)
    info = ck.introspect(sd)
    print(f"[1] introspect ({len(sd)} tensors): {info}")

    cfg = VTWAMConfig(
        video_dit_ckpt=dit_ckpt,
        vae_model_dir=vae_dir,
        device=dev,
        num_views=2,
        proprio_dim=8,
    )
    cfg.apply_introspection(info)
    cfg.objective, cfg.temporal_mode = "ddpm", "factored"
    cfg.time_dist_shift = 1.0
    cfg.action_layers = cfg.num_layers

    enc = VAEVideoEncoder(cfg).to(dev)
    model = VTWAMModel(cfg, vae_encoder=enc).to(dev)

    missing, unexpected = model.video.load_pretrained(sd, strict=False)
    print(f"[2] video DiT load: {len(missing)} missing / {len(unexpected)} unexpected")
    if missing or unexpected:
        raise RuntimeError("video DiT checkpoint did not load cleanly")

    B, T = 1, cfg.n_frames
    video1 = torch.rand(B, 3, T, 256, 256, device=dev)
    video2 = torch.rand(B, 3, T, 256, 256, device=dev)
    z = torch.cat([enc.encode(video1), enc.encode(video2)], dim=3)
    print(f"[3] VAE latent: {tuple(z.shape)} (expect (1, {T}, 32, 64, 16))")
    assert z.shape == (B, T, 32, 64, 16)

    action = torch.randn(B, cfg.chunk_size, cfg.action_dim, device=dev)
    ctx = torch.randn(B, 5, cfg.text_dim, device=dev)
    ctx_mask = torch.ones(B, 5, dtype=torch.bool, device=dev)
    state = torch.randn(B, cfg.proprio_dim, device=dev)
    ctx, ctx_mask = model.build_context(ctx, ctx_mask, state)
    loss, parts = model.training_loss(z, action, ctx, ctx_mask)
    print(f"[4] loss={float(loss.detach()):.4f} {parts}")
    chunk = model.sample_actions(z[:, : cfg.num_history], ctx, ctx_mask)
    print(f"[5] sampled action chunk: {tuple(chunk.shape)}")
    print("VTWAM SERVER VERIFY OK")


if __name__ == "__main__":
    main()
