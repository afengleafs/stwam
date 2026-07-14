"""Roll out a trained STWAM policy in the LIBERO simulator and report success rates.

STWAM is FastWAM-style: the action expert conditions only on the single current
``num_history`` anchor frame (V-JEPA encodes each frame independently, so one
frame suffices), the task language embedding, and the 8-D proprio state.  We
drive the sim closed-loop, replanning a fresh ``chunk_size`` action chunk every
``n_action_steps`` steps (handled inside ``policy.select_action``).

Observation handling mirrors lerobot's ``LiberoProcessorStep`` exactly so the
policy sees its training distribution:
  * images flipped 180 deg (HuggingFaceVLA/libero camera convention),
  * state = [eef_pos(3), quat2axisangle(eef_quat, xyzw)(3), gripper_qpos(2)].

Run (headless server):
    MUJOCO_GL=egl .venv/bin/python eval_libero.py \
        --checkpoint checkpoint/stwam_libero_ddp/latest.pt \
        --suite libero_spatial --n-episodes 10 --device cuda:1
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

# Project modules first (needs project root on sys.path).
from model.config import STWAMConfig
from policy.stwam_policy import STWAMPolicy


PROJECT_ROOT = Path(__file__).resolve().parent
LIBERO_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")
SUITE_MAX_STEPS = {
    "libero_spatial": 280, "libero_object": 280, "libero_goal": 300,
    "libero_10": 520, "libero_90": 400,
}
CAM_AGENT = "agentview_image"
CAM_WRIST = "robot0_eye_in_hand_image"
DUMMY_ACTION = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)


# ---------------------------------------------------------------- libero import
def import_libero():
    """Import the installed ``libero`` package, not the local ``libero/`` dataset dir.

    The dataset directory ``./libero`` is a namespace-package portion that can
    shadow the real package, so drop project-root / cwd entries from sys.path
    for the duration of the import.
    """
    shadow = {os.path.abspath(p) for p in ("", ".", str(PROJECT_ROOT), os.getcwd())}
    saved = list(sys.path)
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") not in shadow]
    try:
        import libero.libero as ll
        from libero.libero import benchmark  # noqa: F401
        from libero.libero.envs import OffScreenRenderEnv  # noqa: F401
    finally:
        sys.path[:] = saved
    return ll


# ------------------------------------------------------------------- text model
class TextEncoder:
    """FLAN-T5 encoder for per-task language embeddings (matches training)."""

    def __init__(self, model_dir: str, device: torch.device, dtype: torch.dtype, max_length: int = 128):
        from transformers import AutoTokenizer, T5EncoderModel

        self.device, self.dtype, self.max_length = device, dtype, max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.encoder = T5EncoderModel.from_pretrained(model_dir, torch_dtype=dtype, local_files_only=True)
        self.encoder.eval().to(device)
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self._cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    @torch.no_grad()
    def __call__(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        if text not in self._cache:
            toks = self.tokenizer(
                [text], padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
            ).to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.device.type == "cuda"):
                hidden = self.encoder(**toks).last_hidden_state.float()
            self._cache[text] = (hidden, toks.attention_mask.bool())
        return self._cache[text]


# ------------------------------------------------------------------- obs helpers
def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """(x, y, z, w) quaternion -> (3,) axis-angle. Mirrors LiberoProcessorStep."""
    quat = np.asarray(quat, dtype=np.float32)
    w = float(np.clip(quat[3], -1.0, 1.0))
    den = float(np.sqrt(max(1.0 - w * w, 0.0)))
    if den < 1e-10:
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] / den * (2.0 * np.arccos(w))).astype(np.float32)


def build_state(raw_obs: dict) -> np.ndarray:
    """[eef_pos(3), quat2axisangle(eef_quat)(3), gripper_qpos(2)] -> (8,)."""
    return np.concatenate([
        np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float32),
        quat2axisangle(raw_obs["robot0_eef_quat"]),
        np.asarray(raw_obs["robot0_gripper_qpos"], dtype=np.float32),
    ], axis=-1)


def flip180(img: np.ndarray) -> np.ndarray:
    """Rotate an HWC image 180 deg (flip H and W), matching LiberoProcessorStep."""
    return np.ascontiguousarray(img[::-1, ::-1, :])


def make_batch(raw_obs: dict, text_embeds: torch.Tensor, text_mask: torch.Tensor,
               device: torch.device, *, no_proprio: bool = False) -> dict[str, Any]:
    """One-frame (T=1) policy batch from a raw LIBERO observation."""
    v1 = torch.from_numpy(flip180(raw_obs[CAM_AGENT]))[None, None].to(device)   # [1,1,H,W,3] uint8
    v2 = torch.from_numpy(flip180(raw_obs[CAM_WRIST]))[None, None].to(device)
    batch = {
        "observation.images.image": v1,
        "observation.images.image2": v2,
        "text_embeds": text_embeds.to(device),
        "text_mask": text_mask.to(device),
    }
    if not no_proprio:
        batch["observation.state"] = torch.from_numpy(build_state(raw_obs))[None].to(device)  # [1,8]
    return batch


# ------------------------------------------------------------------- policy load
def load_policy(checkpoint: str, device: torch.device) -> tuple[STWAMPolicy, STWAMConfig]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = STWAMConfig(**dict(ckpt["config"]))
    cfg.device = str(device)
    policy = STWAMPolicy(cfg).to(device)
    missing, unexpected = policy.load_state_dict(ckpt["policy"], strict=False)
    bad = [k for k in missing if "vjepa" not in k and "backbone" not in k]
    if bad:
        print(f"[policy] WARNING missing trained keys ({len(bad)}): {bad[:8]}")
    if unexpected:
        print(f"[policy] note: {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")
    policy.eval()
    print(f"[policy] loaded step={ckpt.get('step')} chunk={cfg.chunk_size} "
          f"n_action_steps={cfg.n_action_steps} num_views={cfg.num_views}")
    return policy, cfg


# ----------------------------------------------------------------------- sim env
class LiberoSimEnv:
    """Thin closed-loop wrapper around LIBERO's OffScreenRenderEnv.

    reset() settles the scene with no-op steps and forces delta control (the
    training convention); step() reports success via check_success().
    """

    def __init__(self, bddl_file: str, OffScreenRenderEnv, height=256, width=256,
                 init_states=None, num_steps_wait: int = 10):
        self._env = OffScreenRenderEnv(
            bddl_file_name=bddl_file, camera_heights=height, camera_widths=width
        )
        self._env.reset()
        self.num_steps_wait = num_steps_wait
        self._init_states = init_states
        self._init_id = 0

    def _set_delta(self):
        for robot in self._env.env.robots:
            robot.controller.use_delta = True

    def reset(self) -> dict:
        raw = self._env.reset()
        if self._init_states is not None and len(self._init_states):
            raw = self._env.set_init_state(self._init_states[self._init_id % len(self._init_states)])
            self._init_id += 1
        for _ in range(self.num_steps_wait):
            raw, _, _, _ = self._env.step(DUMMY_ACTION)
        self._set_delta()
        return raw

    def step(self, action: np.ndarray) -> tuple[dict, bool, bool]:
        raw, _reward, done, _info = self._env.step(action)
        success = bool(self._env.check_success())
        return raw, (bool(done) or success), success

    def close(self):
        self._env.close()


# ----------------------------------------------------------------------- rollout
def rollout_task(policy, env: LiberoSimEnv, text_embeds, text_mask, device, dtype,
                 n_episodes: int, max_steps: int, *, no_proprio: bool = False) -> tuple[int, list[int]]:
    successes = 0
    ep_steps: list[int] = []
    for ep in range(n_episodes):
        raw_obs = env.reset()
        policy.reset()
        success = False
        steps = 0
        while steps < max_steps:
            batch = make_batch(raw_obs, text_embeds, text_mask, device, no_proprio=no_proprio)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                action = policy.select_action(batch)            # [1, 7]
            action = np.clip(action.squeeze(0).float().cpu().numpy(), -1.0, 1.0).astype(np.float32)
            raw_obs, terminated, success = env.step(action)
            steps += 1
            if terminated:
                break
        successes += int(success)
        ep_steps.append(steps)
        print(f"    ep {ep:02d}: {'SUCCESS' if success else 'fail   '} steps={steps}")
    return successes, ep_steps


# --------------------------------------------------------------------------- cli
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
    p.add_argument("--no-proprio", action="store_true",
                   help="omit observation.state from the policy batch (proprio-free eval)")
    return p.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
            "fp16": torch.float16, "float16": torch.float16,
            "fp32": torch.float32, "float32": torch.float32}[name]


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
                policy, env, text_embeds, text_mask, device, dtype, args.n_episodes, max_steps,
                no_proprio=args.no_proprio,
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
