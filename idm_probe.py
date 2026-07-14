"""P0-3: Inverse-dynamics (IDM) probe of latent action recoverability.

Mirrors Semantic-WM Table 2 at the policy-data level: for each frozen encoder
(STWAM's V-JEPA 2.1 + S-VAE 96-d  vs  V-TWAM's SD3-VAE 16-d) we freeze the
encoder, encode LIBERO clips, and fit a small MLP that regresses the action
segment between two latent frames (z_i, z_{i+k}) -> a[4i : 4(i+k)].  We report
the test-set Pearson r (mean over the 7 action dims).  Higher r == the latent
exposes action-induced change more readily.  This turns Q2 from a success-rate
correlation into a mechanism statement.

The world model is NOT involved (encoder latents only), so no main-model
training is needed.  Same clips, same pairs, same probe capacity for both
encoders -> the only variable is the latent space.

Run:
  CUDA_VISIBLE_DEVICES=4 .venv/bin/python idm_probe.py --device cuda:0 \
      --batches 60 --output logs/idm_probe/idm_results.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from model.config import STWAMConfig
from train_libero import build_fastwam_libero_delta_timestamps


# ------------------------------------------------------------------ data
def build_loader(dataset_root: str, batch_size: int, num_workers: int = 2):
    from torch.utils.data import DataLoader
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata("local/libero", root=dataset_root)
    delta_ts, video_idx, state_idx, action_idx = build_fastwam_libero_delta_timestamps(meta.fps)
    ds = LeRobotDataset("local/libero", root=dataset_root, delta_timestamps=delta_ts, video_backend="pyav")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                        drop_last=True, pin_memory=True)
    # video frames are at raw steps video_idx (0,4,..,32); actions at 0..31.
    stride = video_idx[1] - video_idx[0]  # =4 raw actions per video-frame step
    return loader, video_idx, action_idx, stride


def _canon(video: torch.Tensor) -> torch.Tensor:
    """[B,T,3,H,W] or [B,T,H,W,3], uint8/float -> [B,3,T,H,W] in [0,1]."""
    if video.dtype == torch.uint8:
        video = video.float() / 255.0
    else:
        video = video.float()
        if video.numel() and video.amax() > 2:
            video = video / 255.0
    if video.ndim != 5:
        raise ValueError(f"expected 5D video, got {tuple(video.shape)}")
    if video.shape[2] == 3:        # [B,T,3,H,W]
        return video.permute(0, 2, 1, 3, 4).contiguous()
    if video.shape[-1] == 3:       # [B,T,H,W,3]
        return video.permute(0, 4, 1, 2, 3).contiguous()
    raise ValueError(f"cannot find channel axis in {tuple(video.shape)}")


@torch.no_grad()
def encode_clip(encoder, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """Two views -> latent [B,T,h,w,C] concatenated along width (matches models)."""
    z1 = encoder.encode(v1)
    z2 = encoder.encode(v2)
    return torch.cat([z1, z2], dim=3)


def pool_frames(z: torch.Tensor) -> torch.Tensor:
    """[B,T,h,w,C] -> [B,T,2C] spatial mean+std (keeps some spatial variance)."""
    m = z.mean(dim=(2, 3))
    s = z.std(dim=(2, 3))
    return torch.cat([m, s], dim=-1)


# ------------------------------------------------------------------ probe
class MLP(nn.Module):
    def __init__(self, din: int, dout: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, dout),
        )

    def forward(self, x):
        return self.net(x)


def fit_probe(X: torch.Tensor, Y: torch.Tensor, action_dim: int, device, epochs=400, seed=0):
    """Train/test split, standardize, fit MLP, return mean per-action-dim Pearson r."""
    g = torch.Generator().manual_seed(seed)
    n = X.shape[0]
    perm = torch.randperm(n, generator=g)
    ntr = int(0.8 * n)
    tr, te = perm[:ntr], perm[ntr:]
    Xtr, Xte, Ytr, Yte = X[tr], X[te], Y[tr], Y[te]

    xm, xs = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp_min(1e-6)
    ym, ys = Ytr.mean(0, keepdim=True), Ytr.std(0, keepdim=True).clamp_min(1e-6)
    Xtr, Xte = (Xtr - xm) / xs, (Xte - xm) / xs
    Ytr_n = (Ytr - ym) / ys

    model = MLP(X.shape[1], Y.shape[1]).to(device)
    Xtr, Ytr_n, Xte = Xtr.to(device), Ytr_n.to(device), Xte.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.MSELoss()
    bs = 512
    for ep in range(epochs):
        pi = torch.randperm(Xtr.shape[0], device=device)
        for i in range(0, Xtr.shape[0], bs):
            idx = pi[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(Xtr[idx]), Ytr_n[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = (model(Xte).cpu() * ys + ym)  # de-standardize
    # per-action-dim Pearson r: Y columns are [seg_len, action_dim] flattened
    P = pred.numpy().reshape(pred.shape[0], -1, action_dim)
    T = Yte.numpy().reshape(Yte.shape[0], -1, action_dim)
    rs = []
    for d in range(action_dim):
        p = P[:, :, d].reshape(-1)
        t = T[:, :, d].reshape(-1)
        if p.std() < 1e-8 or t.std() < 1e-8:
            continue
        rs.append(np.corrcoef(p, t)[0, 1])
    return float(np.mean(rs)), int(len(te))


# ------------------------------------------------------------------ collect
@torch.no_grad()
def collect(encoder, loader, n_batches, ks, stride, action_dim, device, dtype):
    """Return {k: (X, Y)} of pooled latent-pair features and action-segment targets."""
    buf = {k: {"X": [], "Y": []} for k in ks}
    it = iter(loader)
    for b in range(n_batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        v1 = _canon(batch["observation.images.image"]).to(device)
        v2 = _canon(batch["observation.images.image2"]).to(device)
        act = batch["action"].float()                      # [B,32,7]
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
            z = encode_clip(encoder, v1, v2)               # [B,T,h,w,C]
        feat = pool_frames(z.float()).cpu()                # [B,T,2C]
        B, T, _ = feat.shape
        for k in ks:
            for i in range(0, T - k):
                a0, a1 = i * stride, (i + k) * stride
                if a1 > act.shape[1]:
                    break
                x = torch.cat([feat[:, i], feat[:, i + k]], dim=-1)     # [B,4C]
                y = act[:, a0:a1].reshape(B, -1)                        # [B, k*stride*7]
                buf[k]["X"].append(x)
                buf[k]["Y"].append(y)
        print(f"  [{encoder.__class__.__name__}] batch {b+1}/{n_batches}", flush=True)
    out = {}
    for k in ks:
        if buf[k]["X"]:
            out[k] = (torch.cat(buf[k]["X"]), torch.cat(buf[k]["Y"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="libero")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--batches", type=int, default=60)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--adapter-ckpt", default="weights/vjepa/adapter_vjepa_image_96.pt")
    ap.add_argument("--vjepa2-ckpt", default="weights/vjepa/vjepa2_1_vitl_dist_vitG_384.pt")
    ap.add_argument("--vae-model-dir", default="vtwam/checkpoint/sd3-medium-diffusers")
    ap.add_argument("--output", default="logs/idm_probe/idm_results.csv")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    action_dim = 7

    loader, video_idx, action_idx, stride = build_loader(args.dataset_root, args.batch_size)
    print(f"[data] video_idx={video_idx} stride={stride} action_len={len(action_idx)}")

    # --- encoders (frozen) ---
    print("[enc] building V-JEPA/S-VAE (STWAM) ...", flush=True)
    scfg = STWAMConfig(adapter_ckpt=args.adapter_ckpt, vjepa2_ckpt=args.vjepa2_ckpt)
    from model.vjepa_encoder import VJEPASemanticEncoder
    vjepa = VJEPASemanticEncoder(scfg).to(device).eval()

    print("[enc] building SD3-VAE (V-TWAM) ...", flush=True)
    from vtwam.config import VTWAMConfig
    from vtwam.vae_encoder import VAEVideoEncoder
    vcfg = VTWAMConfig(vae_model_dir=args.vae_model_dir)
    vcfg.dtype = args.dtype
    vae = VAEVideoEncoder(vcfg).to(device).eval()

    rows = []
    for name, enc in [("STWAM_vjepa_svae96", vjepa), ("VTWAM_sd3vae16", vae)]:
        print(f"\n[collect] {name}", flush=True)
        data = collect(enc, loader, args.batches, args.ks, stride, action_dim, device, dtype)
        for k in args.ks:
            if k not in data:
                continue
            X, Y = data[k]
            r, ntest = fit_probe(X, Y, action_dim, device)
            print(f"[result] {name}  k={k}  IDM Pearson r = {r:.4f}  (N={X.shape[0]}, test={ntest})", flush=True)
            rows.append({"encoder": name, "k": k, "idm_pearson_r": round(r, 4),
                         "n_pairs": int(X.shape[0]), "n_test": ntest})

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["encoder", "k", "idm_pearson_r", "n_pairs", "n_test"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n[done] wrote {args.output}")
    # quick summary
    print("\n=== IDM probe summary (higher r = latent exposes action better) ===")
    for row in rows:
        print(f"  {row['encoder']:22s} k={row['k']}  r={row['idm_pearson_r']}")


if __name__ == "__main__":
    main()
