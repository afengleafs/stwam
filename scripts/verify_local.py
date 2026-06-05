"""Light local verification (CPU, random weights, tiny dims).

Checks the *new* STWAM wiring without needing the big checkpoints/encoders:
  1. coupled forward + training_loss produce a scalar loss that backprops;
  2. zero-init joint adapters => video output is numerically identical to the
     raw vendored DiT (the "load == pretrained" guarantee);
  3. sample_actions returns the right shape.

Run:  python -m scripts.verify_local      (from repo root, in the venv)
"""
import torch

from model.config import STWAMConfig
from model.modeling_stwam import STWAMModel


def main() -> None:
    torch.manual_seed(0)
    cfg = STWAMConfig(
        video_dim=384, in_channels=96, patch_size=1, num_layers=2, num_heads=6,
        wide_head=True, decoder_dim=256, action_dim=7, action_hidden=128,
        action_layers=2, action_heads=4, mot_da=384, mot_heads=6,
        chunk_size=4, text_dim=64, num_history=2, time_dist_shift=2.45,
    )
    model = STWAMModel(cfg).eval()

    B, T, H, W, C = 2, 3, 16, 16, 96
    z = torch.randn(B, T, H, W, C)
    action_chunk = torch.randn(B, cfg.chunk_size, cfg.action_dim)
    video_action = torch.randn(B, T, cfg.action_dim)
    ctx = torch.randn(B, 5, cfg.text_dim)
    ctx_mask = torch.ones(B, 5, dtype=torch.bool)

    # 1. training loss + backward
    loss, parts = model.training_loss(z, action_chunk, video_action, ctx, ctx_mask)
    loss.backward()
    grad_ok = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    print(f"[1] training_loss = {float(loss):.4f}  parts={parts}  backward_grads={grad_ok}")
    assert torch.isfinite(loss) and grad_ok

    # 2. zero-init no-op: coupled video output == raw DiT.forward
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        t = torch.zeros(B, T, dtype=torch.long)
        x_video_t = torch.randn(B, T, H, W, C)
        t_a = torch.rand(B)
        x_a = torch.randn(B, cfg.chunk_size, cfg.action_dim)
        v_video, v_action = model.coupled_forward(x_video_t, t, video_action, x_a, t_a, ctx, ctx_mask)
        ref = model.video.dit(x_video_t, t, video_action)
        max_diff = (v_video - ref).abs().max().item()
    print(f"[2] zero-init no-op: video vs raw DiT max|diff| = {max_diff:.2e}  "
          f"v_video={tuple(v_video.shape)} v_action={tuple(v_action.shape)}")
    assert max_diff < 1e-5, "adapter is not a no-op at init!"

    # 3. sample_actions
    with torch.no_grad():
        anchor = torch.randn(B, cfg.num_history, H, W, C)
        va = torch.zeros(B, cfg.num_history, cfg.action_dim)
        chunk = model.sample_actions(anchor, va, ctx, ctx_mask, num_steps=3)
    print(f"[3] sample_actions -> {tuple(chunk.shape)} (expect ({B}, {cfg.chunk_size}, {cfg.action_dim}))")
    assert chunk.shape == (B, cfg.chunk_size, cfg.action_dim)

    n_params = sum(p.numel() for p in model.parameters())
    n_action = sum(p.numel() for p in model.action.parameters())
    n_adapt = sum(p.numel() for p in model.adapters.parameters())
    print(f"[ok] params total={n_params/1e6:.1f}M  action={n_action/1e6:.2f}M  adapters={n_adapt/1e6:.2f}M")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
