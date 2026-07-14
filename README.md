# STWAM

Parameter-efficient **Semantic-laTent World-Action Model** for language-conditioned manipulation: frozen V-JEPA 2.1 + S-VAE 96-d latents, latent-video DiT co-training, lightweight action expert with MoT adapters. Evaluated on LIBERO / LIBERO-PRO; V-TWAM is the SD3-VAE latent ablation.

## Quick map

| Path | Role |
|---|---|
| [`RUN.md`](RUN.md) | Environment, weights, train / eval commands |
| [`docs/paper/`](docs/paper/) | Results notes, Semantic-WM survey, ICRA draft |
| [`model/`](model/) | STWAM model (vendored semantic-wm under `model/_swm/`) |
| [`policy/`](policy/) | Policy wrapper for LIBERO eval |
| [`vtwam/`](vtwam/) | VAE-latent ablation (same WAM path, SD3 VAE) |
| [`scripts/`](scripts/) | Launch / verify helpers |
| [`artifacts/MANIFEST.md`](artifacts/MANIFEST.md) | Local checkpoints, logs, what to keep or delete |
| `checkpoint/`, `weights/`, `libero/`, `logs/` | Local artifacts (gitignored) |

## Train / eval entrypoints (repo root)

```bash
# STWAM DDP (background wrapper → scripts/)
./train_stwam_libero_ddp_bg.sh

# V-TWAM DDP
./train_vtwam_libero_ddp_bg.sh

# Standard LIBERO eval
MUJOCO_GL=egl .venv/bin/python eval_libero.py \
  --checkpoint checkpoint/stwam_libero_ddp/latest.pt \
  --suite libero_spatial --n-episodes 10 --device cuda:0 \
  --output logs/eval_libero_spatial.csv
```

Full recipes: [`RUN.md`](RUN.md). LIBERO-PRO: [`eval_libero_plus/`](eval_libero_plus/).

## Paper docs

| File | Content |
|---|---|
| [`docs/paper/Paper.md`](docs/paper/Paper.md) | Local CSV + public LIBERO / LIBERO-PRO numbers |
| [`docs/paper/Semantic-WM-Research.md`](docs/paper/Semantic-WM-Research.md) | Survey of Semantic-WM (latent choice evidence) |
| [`docs/paper/STWAM_ICRA2027_draft.md`](docs/paper/STWAM_ICRA2027_draft.md) | ICRA 2027 draft |
| [`docs/paper/figures/`](docs/paper/figures/) | Architecture / paper HTML assets |

Path mapping from older root layout: `Paper.md` → `docs/paper/Paper.md` (same for the other two markdowns).
