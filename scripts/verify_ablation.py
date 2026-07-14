"""Equivalence + smoke checks for model/ablation.py (must pass before training).

1. two-phase equivalence: training_loss_ablation(k=1, p=0, no pooled) must
   reproduce STWAMModel.training_loss bit-for-bit under a fixed seed (fp32),
   and within bf16 tolerance under autocast;
2. pooled no-op: with a zero-init PooledAdaLNCond attached, the ablation loss
   and sample_actions (which dispatches to sample_actions_ablation) must match
   the originals;
3. factor smoke: k=8 and proprio_dropout=0.5 run finite; pooled-"only"
   freezing zeroes every gate_a and sampling still works.
"""
import torch

from eval_libero import load_policy
from model.ablation import (
    PooledAdaLNCond,
    freeze_layerwise_read_path,
    sample_actions_ablation,
    training_loss_ablation,
)

DEV = torch.device("cuda:0")


def _dummy(cfg, device):
    torch.manual_seed(123)
    B, H, W = 2, 16, 16 * cfg.num_views
    z = torch.randn(B, cfg.n_frames, H, W, cfg.in_channels, device=device)
    action = torch.randn(B, cfg.chunk_size, cfg.action_dim, device=device)
    ctx = torch.randn(B, 6, cfg.text_dim, device=device)   # 5 text + 1 "proprio" col
    mask = torch.ones(B, 6, dtype=torch.bool, device=device)
    pad = torch.zeros(B, cfg.chunk_size, dtype=torch.bool, device=device)
    pad[:, -3:] = True
    return z, action, ctx, mask, pad


def main():
    policy, cfg = load_policy("checkpoint/stwam_libero_ddp/latest.pt", DEV)
    m = policy.model
    z, action, ctx, mask, pad = _dummy(cfg, DEV)

    # ---------------- (1) two-phase equivalence -----------------------------
    with torch.no_grad():
        for enabled, tag, lim in [(False, "fp32", 1e-6), (True, "bf16", 1e-2)]:
            with torch.autocast("cuda", torch.bfloat16, enabled=enabled):
                torch.manual_seed(0)
                ref, ref_parts = m.training_loss(z, action, ctx, mask, action_is_pad=pad)
                torch.manual_seed(0)
                abl, abl_parts = training_loss_ablation(
                    m, z, action, ctx, mask, k_draws=1, proprio_dropout=0.0, action_is_pad=pad)
            d = abs(float(ref) - float(abl))
            dv = abs(ref_parts["loss_video"] - abl_parts["loss_video"])
            da = abs(ref_parts["loss_action"] - abl_parts["loss_action"])
            print(f"[equiv:{tag}] |d_total|={d:.3e} |d_video|={dv:.3e} |d_action|={da:.3e}")
            assert d < lim and dv < lim and da < lim, f"{tag} two-phase mismatch"

    # ---------------- (2) pooled zero-init is a no-op ------------------------
    m.pooled_cond = PooledAdaLNCond(cfg.video_dim, cfg.action_hidden, 8).to(DEV)
    with torch.no_grad():
        torch.manual_seed(0)
        ref, _ = m.training_loss(z, action, ctx, mask, action_is_pad=pad)
        torch.manual_seed(0)
        abl, _ = training_loss_ablation(m, z, action, ctx, mask, k_draws=1, action_is_pad=pad)
        assert abs(float(ref) - float(abl)) < 1e-6, "pooled zero-init changed the loss"

        z1 = z[:, : cfg.num_history]
        torch.manual_seed(0)
        m.pooled_cond, saved = None, m.pooled_cond
        a_ref = m.sample_actions(z1, ctx, mask)
        m.pooled_cond = saved
        torch.manual_seed(0)
        a_pool = m.sample_actions(z1, ctx, mask)   # dispatches to ablation path
        d = (a_ref - a_pool).abs().max().item()
        print(f"[pooled-noop] sample_actions max|d|={d:.3e}")
        assert d < 1e-5, "pooled zero-init changed sampling"

    # ---------------- (3) factor smoke ---------------------------------------
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        torch.manual_seed(0)
        l8, p8 = training_loss_ablation(m, z, action, ctx, mask, k_draws=8,
                                        proprio_dropout=0.5, action_is_pad=pad)
        assert torch.isfinite(l8), "k=8 + dropout loss non-finite"
        print(f"[smoke] k=8 p=0.5 loss={float(l8):.4f} "
              f"video={p8['loss_video']:.4f} action={p8['loss_action']:.4f}")

    frozen = freeze_layerwise_read_path(m)
    assert all(float(ad.gate_a.abs().max()) == 0.0 for ad in m.adapters)
    assert all(not ad.projA.weight.requires_grad for ad in m.adapters)
    assert all(ad.projV.weight.requires_grad for ad in m.adapters), "projV must stay trainable (shared with text path)"
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        a_only = sample_actions_ablation(m, z[:, : cfg.num_history], ctx, mask)
        assert a_only.isfinite().all()
    print(f"[pooled-only] froze {len(frozen)} entries; sampling finite. ALL OK")


if __name__ == "__main__":
    main()
