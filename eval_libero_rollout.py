"""LIBERO eval for STWAM with inference-time future rollout (FastWAM-IDM-style).

Same protocol as ``eval_libero.py`` (which it reuses wholesale), plus:
  --future-mode {none,rollout}   none = original sample_actions path
  --action-ctx-frames N          frames the action expert reads (-1 = full
                                 window = cfg.n_frames; 1 = control variant,
                                 should reproduce the baseline)
  --video-steps K                video DDIM steps (0 = cfg.sampling_timesteps)

Run (headless server):
    MUJOCO_GL=egl .venv/bin/python eval_libero_rollout.py \
        --checkpoint checkpoint/stwam_libero_ddp/latest.pt \
        --suite libero_spatial --n-episodes 10 --device cuda:1 \
        --future-mode rollout --action-ctx-frames -1 \
        --output logs/rollout_eval/libero_spatial_rollout_ctx9.csv
"""
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
    LIBERO_SUITES, SUITE_MAX_STEPS, LiberoSimEnv, TextEncoder,
    _dtype, import_libero, load_policy, rollout_task,
)
from model.rollout import sample_actions_rollout
from policy.stwam_policy import STWAMPolicy

PROJECT_ROOT = Path(__file__).resolve().parent


class STWAMRolloutPolicy(STWAMPolicy):
    """STWAMPolicy with the rollout inference path.

    Adds only plain class attributes (no new state), so swapping a loaded
    policy's ``__class__`` to this is safe.
    """

    future_mode: str = "rollout"
    action_ctx_frames: int | None = None   # None -> config.n_frames
    video_steps: int | None = None         # None -> config.sampling_timesteps

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict) -> torch.Tensor:
        z = self._encode_video(batch)
        z_hist = z[:, : self.config.num_history]
        ctx, ctx_mask = self._context(batch)
        if self.future_mode == "none":
            return self.model.sample_actions(z_hist, ctx, ctx_mask)
        return sample_actions_rollout(
            self.model, z_hist, ctx, ctx_mask,
            action_ctx_frames=self.action_ctx_frames,
            video_steps=self.video_steps,
        )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoint/stwam_libero_ddp/latest.pt")
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
    p.add_argument("--future-mode", choices=("none", "rollout"), default="rollout",
                   help="'none' = original direct path, 'rollout' = generate future frames first")
    p.add_argument("--action-ctx-frames", type=int, default=-1,
                   help="frames the action expert reads (-1 = full n_frames window)")
    p.add_argument("--video-steps", type=int, default=0,
                   help="video DDIM steps (0 = cfg.sampling_timesteps)")
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

    policy, cfg = load_policy(args.checkpoint, device)
    policy.__class__ = STWAMRolloutPolicy
    policy.future_mode = args.future_mode
    policy.action_ctx_frames = (args.action_ctx_frames if args.action_ctx_frames > 0
                                else cfg.n_frames)
    policy.video_steps = args.video_steps if args.video_steps > 0 else None
    print(f"[rollout] future_mode={policy.future_mode} "
          f"action_ctx_frames={policy.action_ctx_frames} "
          f"video_steps={policy.video_steps or cfg.sampling_timesteps} "
          f"(n_frames={cfg.n_frames} num_history={cfg.num_history})")

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
                init_path = (Path(libero.get_libero_path("init_states"))
                             / task.problem_folder / task.init_states_file)
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
                "suite": suite_name, "task_id": tid, "language": task.language,
                "success": succ, "episodes": args.n_episodes, "success_rate": rate,
                "mean_steps": float(np.mean(ep_steps)) if ep_steps else 0.0,
                "future_mode": args.future_mode,
                "action_ctx_frames": policy.action_ctx_frames,
            })

    print(f"\n=== OVERALL: {grand_succ}/{grand_total} = {grand_succ/max(grand_total,1)*100:.1f}% ===")
    if args.output and rows:
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[done] wrote {args.output}")


if __name__ == "__main__":
    main()
