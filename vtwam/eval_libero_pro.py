"""Roll out VTWAM on LIBERO-PRO perturbed benchmarks.

This mirrors ``eval_libero_plus/eval_libero_pro.py`` but loads the VAE-latent
VTWAM policy instead of the V-JEPA STWAM policy.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

STWAM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STWAM_ROOT))

from eval_libero import (  # noqa: E402
    LiberoSimEnv,
    TextEncoder,
    _dtype,
    import_libero,
    rollout_task,
)
from eval_libero_plus.eval_libero_pro import (  # noqa: E402
    BASE_SUITES,
    PERTURBATIONS,
    PRO_MAX_STEPS,
    default_evaluation_config,
    ensure_environment_perturbation,
    load_evaluation_config,
    register_libero_pro_objects,
    resolve_benchmark_name,
)
from vtwam.eval_libero import load_policy  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="vtwam/checkpoint/vtwam_libero_ddp/latest.pt")
    p.add_argument("--suite", default="libero_spatial", help="comma-separated base suites")
    p.add_argument("--perturbation", required=True, choices=PERTURBATIONS)
    p.add_argument("--evaluation-config", default=None)
    p.add_argument("--task-ids", type=int, nargs="*", default=None)
    p.add_argument("--n-episodes", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--text-model-dir", default="weights/flan_t5_large")
    p.add_argument("--obs-width", type=int, default=256)
    p.add_argument("--obs-height", type=int, default=256)
    p.add_argument("--no-init-states", action="store_true")
    p.add_argument("--output", default=None)
    return p.parse_args()


def main() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = _dtype(args.dtype)
    eval_cfg = (
        load_evaluation_config(Path(args.evaluation_config))
        if args.evaluation_config
        else default_evaluation_config()
    )

    suites = [s.strip() for s in args.suite.split(",") if s.strip()]
    if args.perturbation == "environment":
        for base_suite in suites:
            ensure_environment_perturbation(base_suite, eval_cfg)

    libero = import_libero()
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    register_libero_pro_objects()
    policy, _cfg = load_policy(str((STWAM_ROOT / args.checkpoint).resolve()), device)
    text_enc = TextEncoder(str((STWAM_ROOT / args.text_model_dir).resolve()), device, dtype)

    bench_dict = benchmark.get_benchmark_dict()
    rows: list[dict[str, Any]] = []
    grand_succ = grand_total = 0

    for base_suite in suites:
        if base_suite not in BASE_SUITES:
            raise ValueError(f"unknown base suite {base_suite!r}; expected one of {BASE_SUITES}")
        bench_name = resolve_benchmark_name(base_suite, args.perturbation)
        if bench_name not in bench_dict:
            raise ValueError(f"benchmark {bench_name!r} not registered; check LIBERO-PRO patch")
        suite = bench_dict[bench_name]()
        task_ids = args.task_ids if args.task_ids is not None else list(range(len(suite.tasks)))
        max_steps = args.max_steps or PRO_MAX_STEPS.get(base_suite, 500)
        print(f"\n=== {bench_name} | perturbation={args.perturbation} | "
              f"{len(task_ids)} task(s) | max_steps={max_steps} ===")

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
                policy,
                env,
                text_embeds,
                text_mask,
                device,
                dtype,
                args.n_episodes,
                max_steps,
            )
            env.close()
            rate = succ / max(args.n_episodes, 1)
            grand_succ += succ
            grand_total += args.n_episodes
            print(f"  task {tid:02d} [{rate*100:5.1f}%] {succ}/{args.n_episodes}"
                  f" | {time.time()-t0:5.1f}s | '{task.language}'")
            rows.append({
                "suite": base_suite,
                "benchmark": bench_name,
                "perturbation": args.perturbation,
                "task_id": tid,
                "language": task.language,
                "success": succ,
                "episodes": args.n_episodes,
                "success_rate": rate,
                "mean_steps": float(np.mean(ep_steps)) if ep_steps else 0.0,
            })

    print(f"\n=== OVERALL ({args.perturbation}): "
          f"{grand_succ}/{grand_total} = {grand_succ/max(grand_total,1)*100:.1f}% ===")
    if args.output and rows:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
