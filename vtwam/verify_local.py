"""Light local VTWAM verification with random weights and random latents."""
from __future__ import annotations

import torch

from .config import VTWAMConfig
from .modeling_vtwam import VTWAMModel


def main() -> None:
    torch.manual_seed(0)
    cfg = VTWAMConfig(
        video_dim=384,
        in_channels=16,
        patch_size=2,
        num_layers=2,
        num_heads=6,
        wide_head=False,
        decoder_dim=256,
        action_dim=7,
        action_hidden=128,
        action_layers=2,
        action_heads=4,
        mot_da=384,
        mot_heads=6,
        chunk_size=4,
        text_dim=64,
        num_history=2,
        time_dist_shift=1.0,
        num_views=2,
        proprio_dim=8,
        vae_model_dir=None,
    )
    model = VTWAMModel(cfg).eval()

    B, T, H, W, C = 2, 3, 32, 64, 16
    z = torch.randn(B, T, H, W, C)
    action_chunk = torch.randn(B, cfg.chunk_size, cfg.action_dim)
    ctx = torch.randn(B, 5, cfg.text_dim)
    ctx_mask = torch.ones(B, 5, dtype=torch.bool)

    loss, parts = model.training_loss(z, action_chunk, ctx, ctx_mask)
    loss.backward()
    grad_ok = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    print(f"[1] training_loss = {float(loss.detach()):.4f} parts={parts} backward_grads={grad_ok}")
    assert torch.isfinite(loss) and grad_ok

    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        t = torch.zeros(B, T, dtype=torch.long)
        x_video_t = torch.randn(B, T, H, W, C)
        t_a = torch.rand(B)
        x_a = torch.randn(B, cfg.chunk_size, cfg.action_dim)
        v_video, v_action = model.coupled_forward(x_video_t, t, x_a, t_a, ctx, ctx_mask)
        xv = model.video.patchify(x_video_t)
        time_cond = model.video.get_time_cond(t)
        for layer in range(model.num_layers):
            xv = model.video.dit.blocks[layer](xv, time_cond, num_views=model.video.dit.num_views)
        ref = model.video.head(xv, time_cond)
        max_diff = (v_video - ref).abs().max().item()
    print(f"[2] zero-init no-op max|diff|={max_diff:.2e} "
          f"v_video={tuple(v_video.shape)} v_action={tuple(v_action.shape)}")
    assert max_diff < 1e-5

    with torch.no_grad():
        anchor = torch.randn(B, cfg.num_history, H, W, C)
        chunk = model.sample_actions(anchor, ctx, ctx_mask, num_steps=3)
    print(f"[3] sample_actions -> {tuple(chunk.shape)}")
    assert chunk.shape == (B, cfg.chunk_size, cfg.action_dim)
    print("VTWAM LOCAL VERIFY OK")


if __name__ == "__main__":
    main()
