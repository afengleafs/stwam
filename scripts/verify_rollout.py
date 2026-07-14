"""Smoke + consistency checks for model/rollout.py (steps a & b of the plan)."""
import torch

from eval_libero import load_policy
from model.rollout import rollout_future_latents, sample_actions_rollout

DEV = torch.device("cuda:1")


def main():
    policy, cfg = load_policy("checkpoint/stwam_libero_ddp/latest.pt", DEV)
    m = policy.model

    H, W = 16, 16 * cfg.num_views
    torch.manual_seed(7)
    z = torch.randn(1, cfg.num_history, H, W, cfg.in_channels, device=DEV)
    ctx = torch.randn(1, 5, cfg.text_dim, device=DEV)
    mask = torch.ones(1, 5, dtype=torch.bool, device=DEV)

    # ---------------- (a) unit smoke under bf16 autocast ----------------
    with torch.autocast("cuda", torch.bfloat16):
        w = rollout_future_latents(m, z, ctx, mask)
        assert w.shape == (1, cfg.n_frames, H, W, cfg.in_channels), w.shape
        assert w.isfinite().all(), "rollout produced non-finite values"
        stds = w.float().std(dim=(0, 2, 3, 4))
        print("[smoke] per-frame std:", [f"{s:.3f}" for s in stds.tolist()])
        a = sample_actions_rollout(m, z, ctx, mask)
        assert a.shape == (1, cfg.chunk_size, cfg.action_dim), a.shape
        assert a.isfinite().all(), "actions non-finite"
    print("[smoke] OK  window", tuple(w.shape), " actions", tuple(a.shape))

    # ------- (b) consistency: rollout+ctx=num_history vs baseline -------
    for enabled, tag in [(False, "fp32"), (True, "bf16")]:
        with torch.autocast("cuda", torch.bfloat16, enabled=enabled):
            torch.manual_seed(0)
            a_ref = m.sample_actions(z, ctx, mask)
            torch.manual_seed(0)
            a_ctl = sample_actions_rollout(m, z, ctx, mask,
                                           action_ctx_frames=cfg.num_history)
            torch.manual_seed(0)
            a_full = sample_actions_rollout(m, z, ctx, mask)
        d_ctl = (a_ref - a_ctl).abs().max().item()
        d_full = (a_ref - a_full).abs().max().item()
        print(f"[consistency:{tag}] ctrl max|d|={d_ctl:.3e}  full max|d|={d_full:.3e}")
        lim = 1e-4 if tag == "fp32" else 1e-2
        assert d_ctl < lim, f"{tag} control diff {d_ctl} exceeds {lim}"
    print("[consistency] OK")


if __name__ == "__main__":
    main()
