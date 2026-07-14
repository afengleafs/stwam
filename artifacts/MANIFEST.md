# Local artifacts manifest

Large files are **gitignored**. This list is for humans: what backs paper tables, and what is safe to delete after confirmation.

Repo footprint (approx.): ~65G total — `checkpoint/` ~36G, `weights/` ~12G, `.venv/` ~10G, `vtwam/checkpoint/` ~6G, `libero/` ~2G.

## Keep (paper / main results)

### Checkpoints

| Path | Use |
|---|---|
| `checkpoint/stwam_libero_ddp/latest.pt` | Main STWAM eval (same as 300k) |
| `checkpoint/stwam_libero_ddp/step_00300000.pt` | Canonical 300k snapshot |
| `checkpoint/stwam_libero_ddp/step_00100000.pt` | Optional step curve |
| `checkpoint/stwam_libero_ddp/step_00200000.pt` | Optional step curve |
| `vtwam/checkpoint/vtwam_libero_ddp/step_00300000.pt` | V-TWAM ablation (Table II) |
| `checkpoint/ablation_kstudy_k1/` | k-draws smoke (Table V) |
| `checkpoint/ablation_kstudy_k8/` | k-draws smoke (Table V) |
| `vtwam/checkpoint/vae/`, `vtwam/checkpoint/sd3-medium-diffusers/` | V-TWAM encoder assets |

### Eval logs (cited in `docs/paper/Paper.md`)

| Path | Use |
|---|---|
| `logs/eval_libero_{spatial,object,goal,10}.csv` | Standard LIBERO STWAM |
| `logs/vtwam_eval/libero/`, `logs/vtwam_eval/summary/` | V-TWAM LIBERO |
| `logs/vtwam_eval/libero_pro/`, `logs/vtwam_eval/summary/libero_pro_*` | V-TWAM PRO |
| `eval_libero_plus/results/libero_pro_matrix.csv` | STWAM LIBERO-PRO matrix |
| `eval_libero_plus/results/libero_pro_summary.md` | STWAM PRO summary |
| `logs/ablation_eval/kstudy_k{1,8}/` | k-draws full protocol |

### Weights / data

| Path | Use |
|---|---|
| `weights/vjepa/` | V-JEPA 2.1, DiT-S_D96, S-VAE adapter |
| `libero/` | LeRobot-format LIBERO demos for training |

## Delete candidates (confirm before rm)

Not used in `docs/paper/Paper.md` main tables. Deleting frees ~18G+.

| Path | ~Size | Note |
|---|---:|---|
| `checkpoint/ablation_r0_control/` | ~9.3G | Internal ablation; not in paper main results |
| `checkpoint/ablation_r1_pdrop05/` | ~9.3G | Same |
| `checkpoint/stwam_libero_ddp/step_00000002.pt` | ~1.9G | Early smoke save |
| `logs/*.pid` | tiny | Stale process pid files |

Do **not** delete keep-list checkpoints or eval CSVs without updating paper numbers.

## Not for git

See root `.gitignore`: `checkpoint/`, `weights/`, `libero/`, `logs/`, `*.pt`, `.venv/`, etc. Tracked summaries only under `eval_libero_plus/results/` and this `MANIFEST.md`.
