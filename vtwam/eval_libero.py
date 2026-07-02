"""Roll out a trained VTWAM policy in the LIBERO simulator."""
from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from eval_libero import (
    LIBERO_SUITES,
    SUITE_MAX_STEPS,
    LiberoSimEnv,
    TextEncoder,
    _dtype,
    import_libero,
    make_batch,
    rollout_task,
)

from .config import VTWAMConfig
from .policy import VTWAMPolicy


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_policy(checkpoint: str, device: torch.device) -> tuple[VTWAMPolicy, VTWAMConfig]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = VTWAMConfig(**dict(ckpt["config"]))
    cfg.device = str(device)
    policy = VTWAMPolicy(cfg).to(device)
    missing, unexpected = policy.load_state_dict(ckpt["policy"], strict=False)
    bad = [k for k in missing if not k.startswith("model.vae.")]
    if bad:
        print(f"[policy] WARNING missing trained keys ({len(bad)}): {bad[:8]}")
    if unexpected:
        print(f"[policy] note: {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")
    policy.eval()
    print(f"[policy] loaded step={ckpt.get('step')} chunk={cfg.chunk_size} "
          f"n_action_steps={cfg.n_action_steps} num_views={cfg.num_views}")
    return policy, cfg


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="vtwam/checkpoint/vtwam_libero_ddp/latest.pt")
    p.add_argument("--suite", default="libero_spatial",
                   help="comma-separated suite(s): " + ", ".join(LIBERO_SUITES))
    p.add_argument("--task-ids", type=int, nargs="*", default=None,
                   help="restrict to these task ids within each suite (default: all)")
    p.add_argument("--n-episodes", type=int, default=10, help="rollouts per task")
    p.add_argument("--max-steps", type=int, default=None, help="override per-episode cap")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--text-model-dir", default="weights/flan_t5_large")
    p.add_argument("--obs-width", type=int, default=256)
    p.add_argument("--obs-height", type=int, default=256)
    p.add_argument("--no-init-states", action="store_true",
                   help="do not load task init states (random reset instead)")
    p.add_argument("--output", default=None, help="optional CSV path for per-task results")
    return p.parse_args()


def main():
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = _dtype(args.dtype)

    libero = import_libero()
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    policy, _cfg = load_policy(args.checkpoint, device)
    text_enc = TextEncoder(str((PROJECT_ROOT / args.text_model_dir).resolve()), device, dtype)

    bench_dict = benchmark.get_benchmark_dict()
    suites = [s.strip() for s in args.suite.split(",") if s.strip()]
    rows: list[dict[str, Any]] = []
    grand_succ = grand_total = 0

    for suite_name in suites:
        if suite_name not in bench_dict:
            raise ValueError(f"unknown suite {suite_name!r}; available: {sorted(bench_dict)}")
        suite = bench_dict[suite_name]()
        task_ids = args.task_ids if args.task_ids is not None else list(range(len(suite.tasks)))
        max_steps = args.max_steps or SUITE_MAX_STEPS.get(suite_name, 500)
        print(f"\n=== suite {suite_name} | {len(task_ids)} task(s) | max_steps={max_steps} ===")

        for tid in task_ids:
            task = suite.get_task(tid)
            bddl = os.path.join(libero.get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            init_states = None
            if not args.no_init_states:
                init_path = (
                    Path(libero.get_libero_path("init_states"))
                    / task.problem_folder
                    / task.init_states_file
                )
                init_states = torch.load(init_path, weights_only=False)
            env = LiberoSimEnv(bddl, OffScreenRenderEnv, args.obs_height, args.obs_width, init_states)

            text_embeds, text_mask = text_enc(task.language)
            t0 = time.time()
            succ, ep_steps = rollout_task(
                policy, env, text_embeds, text_mask, device, dtype, args.n_episodes, max_steps
            )
            env.close()
            rate = succ / max(args.n_episodes, 1)
            grand_succ += succ
            grand_total += args.n_episodes
            print(f"  task {tid:02d} [{rate*100:5.1f}%] {succ}/{args.n_episodes}"
                  f" | {time.time()-t0:5.1f}s | '{task.language}'")
            rows.append({
                "suite": suite_name,
                "task_id": tid,
                "language": task.language,
                "success": succ,
                "episodes": args.n_episodes,
                "success_rate": rate,
                "mean_steps": float(np.mean(ep_steps)) if ep_steps else 0.0,
            })

    print(f"\n=== OVERALL: {grand_succ}/{grand_total} = {grand_succ/max(grand_total,1)*100:.1f}% ===")
    if args.output and rows:
        with open(args.output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[done] wrote {args.output}")


if __name__ == "__main__":
    main()
