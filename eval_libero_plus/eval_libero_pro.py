"""Roll out STWAM on LIBERO-PRO perturbed benchmarks.

Extends [eval_libero.py](../eval_libero.py) with perturbation-aware suite resolution.
Pre-built HF perturbations (object/swap/language/task) use benchmark keys like
``libero_spatial_object``.  Environment perturbation is generated on demand via
``perturbation.create_env`` and renamed from ``*_temp`` to ``*_env``.

Example:
    MUJOCO_GL=egl .venv/bin/python eval_libero_plus/eval_libero_pro.py \\
        --checkpoint checkpoint/stwam_libero_ddp/latest.pt \\
        --suite libero_spatial --perturbation object \\
        --n-episodes 10 --device cuda:1 \\
        --output eval_libero_plus/logs/eval_pro_libero_spatial_object.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

STWAM_ROOT = Path(__file__).resolve().parent.parent
EVAL_PLUS_ROOT = Path(__file__).resolve().parent
LIBERO_PRO_ROOT = EVAL_PLUS_ROOT / "LIBERO-PRO"
LIBERO_PRO_ASSETS = LIBERO_PRO_ROOT / "libero" / "libero" / "assets"
sys.path.insert(0, str(STWAM_ROOT))

from eval_libero import (  # noqa: E402
    TextEncoder,
    LiberoSimEnv,
    _dtype,
    import_libero,
    load_policy,
    rollout_task,
)

BASE_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
PERTURBATIONS = ("object", "swap", "language", "task", "environment")
PERTURB_SUFFIX = {
    "object": "object",
    "swap": "swap",
    "language": "lan",
    "task": "task",
    "environment": "env",
}
PRO_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}
ORI_RATES = {
    "libero_spatial": 0.880,
    "libero_object": 0.980,
    "libero_goal": 0.900,
    "libero_10": 0.830,
}


def register_libero_pro_objects() -> None:
    """Register LIBERO-PRO-only objects into the installed LIBERO runtime.

    The local LIBERO-PRO source tree is not directly import-compatible with this
    Python env: its top-level package is a namespace package, while the installed
    LIBERO package is a regular package.  We therefore keep the known-good
    installed runtime and add only the missing PRO object classes, pointing their
    XML paths at the local LIBERO-PRO assets.
    """
    from libero.libero.envs.base_object import (  # noqa: WPS433
        OBJECTS_DICT,
        VISUAL_CHANGE_OBJECTS_DICT,
        register_object,
        register_visual_change_object,
    )
    from robosuite.models.objects import MujocoXMLObject  # noqa: WPS433

    def asset_path(*parts: str) -> str:
        path = LIBERO_PRO_ASSETS.joinpath(*parts)
        if not path.exists():
            raise FileNotFoundError(f"missing LIBERO-PRO asset: {path}")
        return str(path)

    def category_name(cls_name: str) -> str:
        return "_".join(re.sub(r"([A-Z])", r" \1", cls_name).split()).lower()

    default_joints = object()

    class ProArticulatedObject(MujocoXMLObject):
        def __init__(self, name, obj_name, joints=default_joints):
            if joints is default_joints:
                joints = [dict(type="free", damping="0.0005")]
            super().__init__(
                asset_path("articulated_objects", f"{obj_name}.xml"),
                name=name,
                joints=joints,
                obj_type="all",
                duplicate_collision_geoms=False,
            )
            self.category_name = category_name(self.__class__.__name__)
            self.rotation = (np.pi / 4, np.pi / 2)
            self.rotation_axis = "x"
            self.object_properties = {
                "articulation": {
                    "default_open_ranges": [],
                    "default_close_ranges": [],
                },
                "vis_site_names": {},
            }

    class YellowCabinet(ProArticulatedObject):
        def __init__(self, name="yellow_cabinet", obj_name="yellow_cabinet", joints=default_joints):
            super().__init__(name, obj_name, joints)
            self.object_properties["articulation"]["default_open_ranges"] = [-0.16, -0.14]
            self.object_properties["articulation"]["default_close_ranges"] = [0.0, 0.005]

        def is_open(self, qpos):
            return qpos < max(self.object_properties["articulation"]["default_open_ranges"])

        def is_close(self, qpos):
            return qpos > min(self.object_properties["articulation"]["default_close_ranges"])

    class YellowStove(ProArticulatedObject):
        def __init__(self, name="yellow_stove", obj_name="yellow_stove", joints=default_joints):
            super().__init__(name, obj_name, joints)
            self.rotation = (0, 0)
            self.rotation_axis = "y"
            self.object_properties["vis_site_names"]["burner"] = (
                self.naming_prefix + "burner",
                False,
            )
            self.object_properties["articulation"]["default_turnon_ranges"] = [0.5, 2.1]
            self.object_properties["articulation"]["default_turnoff_ranges"] = [-0.005, 0.0]

        def turn_on(self, qpos):
            is_on = qpos >= min(self.object_properties["articulation"]["default_turnon_ranges"])
            self.object_properties["vis_site_names"]["burner"] = (
                self.naming_prefix + "burner",
                is_on,
            )
            return is_on

        def turn_off(self, qpos):
            is_off = qpos < max(self.object_properties["articulation"]["default_turnoff_ranges"])
            self.object_properties["vis_site_names"]["burner"] = (
                self.naming_prefix + "burner",
                not is_off,
            )
            return is_off

    class ProTurbosquidObject(MujocoXMLObject):
        def __init__(self, name, obj_name, joints=default_joints):
            if joints is default_joints:
                joints = [dict(type="free", damping="0.0005")]
            super().__init__(
                asset_path("turbosquid_objects", obj_name, f"{obj_name}.xml"),
                name=name,
                joints=joints,
                obj_type="all",
                duplicate_collision_geoms=False,
            )
            self.category_name = category_name(self.__class__.__name__)
            self.rotation = (0, 0)
            self.rotation_axis = "x"
            self.object_properties = {"vis_site_names": {}}

    class BrownRack(ProTurbosquidObject):
        def __init__(self, name="brown_rack", obj_name="brown_rack", joints=default_joints):
            super().__init__(name, obj_name, joints)

    class WhiteBottle(ProTurbosquidObject):
        def __init__(self, name="white_bottle", obj_name="white_bottle", joints=default_joints):
            super().__init__(name, obj_name, joints)

    class YellowMokaPot(ProTurbosquidObject):
        def __init__(self, name="yellow_moka_pot", obj_name="yellow_moka_pot", joints=default_joints):
            super().__init__(name, obj_name, joints)

    registered: list[str] = []

    def maybe_register(key: str, cls: type, visual: bool = False) -> None:
        if key not in OBJECTS_DICT:
            register_object(cls)
            registered.append(key)
        if visual and key not in VISUAL_CHANGE_OBJECTS_DICT:
            register_visual_change_object(cls)

    maybe_register("yellow_cabinet", YellowCabinet)
    maybe_register("yellow_stove", YellowStove, visual=True)
    maybe_register("brown_rack", BrownRack)
    maybe_register("white_bottle", WhiteBottle)
    maybe_register("yellow_moka_pot", YellowMokaPot)

    if registered:
        print(f"[pro] registered missing LIBERO-PRO objects: {', '.join(registered)}")


def load_evaluation_config(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def default_evaluation_config() -> dict[str, Any]:
    libero_root = LIBERO_PRO_ROOT / "libero" / "libero"
    return {
        "bddl_files_path": str(libero_root / "bddl_files"),
        "script_path": str(LIBERO_PRO_ROOT / "notebooks" / "generate_init_states.py"),
        "init_file_dir": str(libero_root / "init_files"),
        "use_environment": False,
        "use_swap": False,
        "use_object": False,
        "use_language": False,
        "use_task": False,
        "ood_task_configs": {
            "environment": str(LIBERO_PRO_ROOT / "libero_ood" / "ood_environment.yaml"),
            "swap": str(LIBERO_PRO_ROOT / "libero_ood" / "ood_spatial_relation.yaml"),
            "object": str(LIBERO_PRO_ROOT / "libero_ood" / "ood_object.yaml"),
            "language": str(LIBERO_PRO_ROOT / "libero_ood" / "ood_language.yaml"),
            "task": str(LIBERO_PRO_ROOT / "libero_ood" / "ood_task.yaml"),
        },
        "perturbation_mapping": {
            "use_environment": "env",
            "use_swap": "swap",
            "use_object": "object",
            "use_language": "lan",
            "use_task": "task",
        },
    }


def resolve_benchmark_name(base_suite: str, perturbation: str) -> str:
    suffix = PERTURB_SUFFIX[perturbation]
    return f"{base_suite}_{suffix}"


def ensure_environment_perturbation(base_suite: str, eval_cfg: dict[str, Any]) -> None:
    """Generate libero_*_env bddl/init via LIBERO-PRO if missing."""
    env_dir = Path(eval_cfg["init_file_dir"]) / f"{base_suite}_env"
    if env_dir.exists() and any(env_dir.glob("*.pruned_init")):
        print(f"[pro] environment perturbation already present: {env_dir}")
        return

    temp_bddl = Path(eval_cfg["bddl_files_path"]) / f"{base_suite}_temp"
    temp_init = Path(eval_cfg["init_file_dir"]) / f"{base_suite}_temp"
    env_bddl = Path(eval_cfg["bddl_files_path"]) / f"{base_suite}_env"

    cfg = dict(eval_cfg)
    cfg["bddl_files_path"] = str(Path(eval_cfg["bddl_files_path"]) / base_suite)
    cfg["task_suite_name"] = base_suite
    cfg["use_environment"] = True
    cfg["use_swap"] = False
    cfg["use_object"] = False
    cfg["use_language"] = False
    cfg["use_task"] = False
    cfg["seed"] = 42

    for p in (temp_bddl, temp_init, env_bddl, env_dir):
        if p.exists():
            shutil.rmtree(p)

    sys.path.insert(0, str(LIBERO_PRO_ROOT))
    import perturbation  # noqa: WPS433

    print(f"[pro] generating environment perturbation for {base_suite} ...")
    perturbation.create_env(configs=cfg)

    if not temp_bddl.exists():
        raise RuntimeError(f"expected temp bddl dir missing: {temp_bddl}")
    if not temp_init.exists():
        raise RuntimeError(f"expected temp init dir missing: {temp_init}")

    shutil.move(str(temp_bddl), str(env_bddl))
    shutil.move(str(temp_init), str(env_dir))
    print(f"[pro] environment perturbation ready: {env_bddl}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoint/stwam_libero_ddp/latest.pt")
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
    p.add_argument("--no-proprio", action="store_true",
                   help="omit observation.state from the policy batch (proprio-free eval)")
    return p.parse_args()


def main():
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

    if args.perturbation == "environment":
        for base_suite in [s.strip() for s in args.suite.split(",") if s.strip()]:
            ensure_environment_perturbation(base_suite, eval_cfg)

    libero = import_libero()
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    register_libero_pro_objects()

    ckpt_path = str((STWAM_ROOT / args.checkpoint).resolve())
    policy, _cfg = load_policy(ckpt_path, device)
    text_enc = TextEncoder(str((STWAM_ROOT / args.text_model_dir).resolve()), device, dtype)

    bench_dict = benchmark.get_benchmark_dict()
    suites = [s.strip() for s in args.suite.split(",") if s.strip()]
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
        ori_rate = ORI_RATES.get(base_suite, 0.0)
        print(f"\n=== {bench_name} | perturbation={args.perturbation} | "
              f"{len(task_ids)} task(s) | max_steps={max_steps} ===")

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
            retention = (rate / ori_rate) if ori_rate > 0 else 0.0
            grand_succ += succ
            grand_total += args.n_episodes
            print(f"  task {tid:02d} [{rate*100:5.1f}%] {succ}/{args.n_episodes}"
                  f" | retention={retention*100:5.1f}% | {time.time()-t0:5.1f}s | '{task.language}'")
            rows.append({
                "suite": base_suite,
                "benchmark": bench_name,
                "perturbation": args.perturbation,
                "task_id": tid,
                "language": task.language,
                "success": succ,
                "episodes": args.n_episodes,
                "success_rate": rate,
                "ori_success_rate": ori_rate,
                "retention_vs_ori": retention,
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
