"""Distributed STWAM training and max global batch-size probing.

This entrypoint mirrors ``train_libero.py`` and uses the same FastWAM-style
LIBERO timestamps.  ``--batch-size`` is the global DDP micro-batch size; the
per-rank batch is derived as ``batch_size // world_size``.

Examples:
    CUDA_VISIBLE_DEVICES=2,3,4,5 .venv/bin/python -m torch.distributed.run \
      --standalone --nproc_per_node=4 train_ddp.py \
      --dataset-root libero --batch-size 4 --grad-accum-steps 4 --max-steps 2

    .venv/bin/python train_ddp.py --probe-max-batch \
      --dataset-root libero --probe-gpus 2,3,4,5
"""
from __future__ import annotations

import argparse
import math
import os
import random
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from model import checkpoint as ck
from policy.stwam_policy import STWAMPolicy
from train import (
    _attach_text,
    _build_text_cache,
    _dtype,
    _move_batch,
    _require_imports,
    _task_dict_from_metadata,
    save_checkpoint,
)
from train_libero import (
    _adapt_fastwam_lerobot_batch,
    _make_config,
    build_fastwam_libero_delta_timestamps,
)


OOM_MARKERS = (
    "out of memory",
    "cuda error: out of memory",
    "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed",
    "cuda error: an illegal memory access",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", default="libero")
    p.add_argument("--video-dit-ckpt", default="weights/vjepa/DiT-S_D96.pt")
    p.add_argument("--adapter-ckpt", default="weights/vjepa/adapter_vjepa_image_96.pt")
    p.add_argument("--vjepa2-ckpt", default="weights/vjepa/vjepa2_1_vitl_dist_vitG_384.pt")
    p.add_argument("--text-model-id", default="google/flan-t5-large")
    p.add_argument("--text-model-dir", default="weights/flan_t5_large")
    p.add_argument("--text-cache-path", default=None)
    p.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    p.add_argument("--output-dir", default="checkpoint/stwam_libero_ddp")
    p.add_argument("--resume", default=None)
    p.add_argument("--fastwam-num-frames", type=int, default=33)
    p.add_argument("--fastwam-action-video-freq-ratio", type=int, default=4)
    p.add_argument("--fastwam-global-sample-stride", type=int, default=1)
    p.add_argument("--n-frames", type=int, default=9)
    p.add_argument("--num-history", type=int, default=1)
    p.add_argument("--chunk-size", type=int, default=32)
    p.add_argument("--n-action-steps", type=int, default=32)
    p.add_argument("--num-views", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=4,
                   help="Global batch size. Per-rank batch is batch_size / world_size.")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=300000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--grad-accum-steps", type=int, default=4)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=100000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max-text-length", type=int, default=128)
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--find-unused-parameters", action="store_true")

    p.add_argument("--probe-max-batch", action="store_true",
                   help="Run parent-process max global batch-size search.")
    p.add_argument("--probe-gpus", default="2,3,4,5",
                   help="Physical GPU ids for probe child processes.")
    p.add_argument("--probe-steps", type=int, default=2)
    p.add_argument("--probe-start-global-batch", type=int, default=None)
    p.add_argument("--probe-max-global-batch", type=int, default=192)
    p.add_argument("--probe-candidate-batch", type=int, default=None,
                   help=argparse.SUPPRESS)
    return p.parse_args()


def _rank0(rank: int, *items: Any) -> None:
    if rank == 0:
        print(*items, flush=True)


def _init_distributed() -> tuple[int, int, int, bool]:
    if "LOCAL_RANK" not in os.environ:
        return 0, 0, 1, False
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size, True


def _cleanup_distributed(distributed: bool) -> None:
    if distributed and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _dist_barrier() -> None:
    if torch.cuda.is_available():
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()


def _freeze_unused_train_path(policy: STWAMPolicy) -> None:
    """Freeze known trainable parameters that STWAM's coupled path never reads."""
    action_embedder = getattr(policy.model.video.dit, "action_embedder", None)
    if action_embedder is not None:
        action_embedder.requires_grad_(False)


def _build_loader(args, rank: int, world_size: int, distributed: bool, device):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    dataset_metadata = LeRobotDatasetMetadata("local/libero", root=args.dataset_root)
    delta_timestamps, video_indices, state_indices, action_indices = build_fastwam_libero_delta_timestamps(
        dataset_metadata.fps,
        num_frames=args.fastwam_num_frames,
        action_video_freq_ratio=args.fastwam_action_video_freq_ratio,
        global_sample_stride=args.fastwam_global_sample_stride,
    )
    dataset = LeRobotDataset(
        "local/libero", root=args.dataset_root, delta_timestamps=delta_timestamps,
        video_backend="pyav",
    )
    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
        if distributed else None
    )
    loader = DataLoader(
        dataset,
        batch_size=args.per_rank_batch,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    return dataset_metadata, dataset, sampler, loader, delta_timestamps, video_indices, state_indices, action_indices


def _build_or_load_text_cache_ranked(args, tasks, device, dtype, rank: int, distributed: bool):
    if rank == 0:
        text_embeds, text_mask = _build_text_cache(args, tasks, device, dtype)
    else:
        text_embeds = text_mask = None
    if distributed:
        _dist_barrier()
    if rank != 0:
        text_embeds, text_mask = _build_text_cache(args, tasks, device, dtype)
    return text_embeds, text_mask


def _peak_memory_payload(device, rank: int):
    if device.type != "cuda":
        return {"rank": rank, "allocated_gb": 0.0, "reserved_gb": 0.0}
    gb = 1024 ** 3
    return {
        "rank": rank,
        "allocated_gb": torch.cuda.max_memory_allocated(device) / gb,
        "reserved_gb": torch.cuda.max_memory_reserved(device) / gb,
    }


def train_main(args) -> None:
    local_rank, rank, world_size, distributed = _init_distributed()
    try:
        _require_imports()
        if args.probe_candidate_batch is not None:
            args.batch_size = args.probe_candidate_batch
            args.no_save = True
        if args.batch_size <= 0:
            raise ValueError("--batch-size must be positive")
        if args.batch_size % world_size != 0:
            raise ValueError(
                f"global --batch-size={args.batch_size} must be divisible by world_size={world_size}"
            )
        args.world_size = world_size
        args.distributed = distributed
        args.per_rank_batch = args.batch_size // world_size
        if args.per_rank_batch < 1:
            raise ValueError(
                f"global --batch-size={args.batch_size} gives per-rank batch < 1 "
                f"for world_size={world_size}"
            )

        random.seed(args.seed + rank)
        torch.manual_seed(args.seed + rank)
        if distributed:
            if not torch.cuda.is_available():
                raise RuntimeError("DDP training requires CUDA")
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
        dtype = _dtype(args.dtype)

        for path in (args.video_dit_ckpt, args.adapter_ckpt, args.vjepa2_ckpt):
            if not Path(path).is_file():
                raise FileNotFoundError(path)

        sd = ck.load_raw_state_dict(args.video_dit_ckpt)
        info = ck.introspect(sd)
        dataset_metadata, dataset, sampler, loader, delta_timestamps, video_indices, state_indices, action_indices = _build_loader(
            args, rank, world_size, distributed, device
        )
        cfg = _make_config(args, info, video_frame_count=len(video_indices), action_count=len(action_indices))

        tasks = _task_dict_from_metadata(dataset_metadata)
        _rank0(
            rank,
            f"[ddp] world_size={world_size} global_batch={args.batch_size} "
            f"per_rank_batch={args.per_rank_batch}",
        )
        _rank0(rank, f"[data] frames={len(dataset)} tasks={len(tasks)} fps={dataset_metadata.fps}")
        _rank0(rank, f"[data] video_indices={video_indices}")
        _rank0(rank, f"[data] state_indices={state_indices[0]}..{state_indices[-1]} len={len(state_indices)}")
        _rank0(rank, f"[data] action_indices={action_indices[0]}..{action_indices[-1]} len={len(action_indices)}")
        _rank0(rank, f"[data] delta_timestamps={delta_timestamps}")
        if args.max_steps == 0:
            _rank0(rank, "[done] max_steps=0; validated dataset timestamps only")
            return
        text_embeds, text_mask = _build_or_load_text_cache_ranked(args, tasks, device, dtype, rank, distributed)
        _rank0(rank, f"[text] cache embeds={tuple(text_embeds.shape)} mask={tuple(text_mask.shape)}")

        policy = STWAMPolicy(cfg).to(device)
        missing, unexpected = policy.model.video.load_pretrained(sd, strict=False)
        _rank0(rank, f"[video DiT] load: {len(missing)} missing / {len(unexpected)} unexpected")
        if missing or unexpected:
            raise RuntimeError("video DiT checkpoint did not load cleanly")
        _freeze_unused_train_path(policy)

        if distributed:
            policy = DistributedDataParallel(
                policy,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
                find_unused_parameters=args.find_unused_parameters,
            )
            policy_no_ddp = policy.module
        else:
            policy_no_ddp = policy

        optim_params = [p for p in policy_no_ddp.get_optim_params() if p.requires_grad]
        optimizer = torch.optim.AdamW(optim_params, lr=args.lr, weight_decay=args.weight_decay)

        def lr_lambda(step: int) -> float:
            if step < args.warmup_steps:
                return max(step, 1) / max(args.warmup_steps, 1)
            progress = (step - args.warmup_steps) / max(args.max_steps - args.warmup_steps, 1)
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        start_step = 0
        if args.resume:
            ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
            policy_no_ddp.load_state_dict(ckpt["policy"], strict=True)
            optimizer.load_state_dict(ckpt["optimizer"])
            if ckpt.get("scheduler") is not None:
                scheduler.load_state_dict(ckpt["scheduler"])
            start_step = int(ckpt.get("step", 0))
            _rank0(rank, f"[resume] {args.resume} step={start_step}")

        policy.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        out_dir = Path(args.output_dir)
        step = start_step
        optimizer.zero_grad(set_to_none=True)
        steps_per_epoch = max(len(loader), 1)
        epoch = start_step // steps_per_epoch
        pbar = tqdm(total=args.max_steps, initial=step, desc="train", disable=rank != 0)

        while step < args.max_steps:
            if sampler is not None:
                sampler.set_epoch(epoch)
            for batch in loader:
                if step >= args.max_steps:
                    break
                batch = _adapt_fastwam_lerobot_batch(batch)
                batch = _attach_text(batch, text_embeds, text_mask)
                batch = _move_batch(batch, device)
                should_sync = (step + 1) % args.grad_accum_steps == 0
                sync_context = nullcontext()
                if distributed and not should_sync:
                    sync_context = policy.no_sync()
                with sync_context:
                    with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                        loss, parts = policy(batch)
                        loss = loss / args.grad_accum_steps
                    loss.backward()
                if should_sync:
                    torch.nn.utils.clip_grad_norm_(policy_no_ddp.parameters(), args.grad_clip)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                step += 1
                if rank == 0:
                    pbar.update(1)
                if rank == 0 and step % args.log_every == 0:
                    lr = scheduler.get_last_lr()[0]
                    pbar.write(
                        f"step={step} loss={float(loss.detach()) * args.grad_accum_steps:.4f} "
                        f"video={parts['loss_video']:.4f} action={parts['loss_action']:.4f} lr={lr:.2e}"
                    )
                if rank == 0 and not args.no_save and step % args.save_every == 0:
                    save_checkpoint(out_dir / f"step_{step:08d}.pt", policy_no_ddp, optimizer, scheduler, step, cfg, args)
            epoch += 1

        if rank == 0:
            pbar.close()
            if not args.no_save:
                save_checkpoint(out_dir / "latest.pt", policy_no_ddp, optimizer, scheduler, step, cfg, args)
                print(f"[done] saved {out_dir / 'latest.pt'}", flush=True)

        if args.probe_candidate_batch is not None:
            peak = _peak_memory_payload(device, rank)
            print(
                f"[probe-rank] global_batch={args.batch_size} rank={rank} "
                f"allocated_gb={peak['allocated_gb']:.3f} reserved_gb={peak['reserved_gb']:.3f}",
                flush=True,
            )
            if rank == 0:
                print(
                    f"[probe-pass] global_batch={args.batch_size} per_rank_batch={args.per_rank_batch} "
                    f"steps={args.max_steps}",
                    flush=True,
                )
    finally:
        _cleanup_distributed(distributed)


def _parse_probe_gpus(spec: str) -> list[str]:
    gpus = [item.strip() for item in spec.split(",") if item.strip()]
    if not gpus:
        raise ValueError("--probe-gpus must contain at least one GPU id")
    return gpus


def _child_args(args, candidate: int) -> list[str]:
    child = [
        "--dataset-root", args.dataset_root,
        "--video-dit-ckpt", args.video_dit_ckpt,
        "--adapter-ckpt", args.adapter_ckpt,
        "--vjepa2-ckpt", args.vjepa2_ckpt,
        "--text-model-id", args.text_model_id,
        "--text-model-dir", args.text_model_dir,
        "--hf-endpoint", args.hf_endpoint,
        "--output-dir", args.output_dir,
        "--fastwam-num-frames", str(args.fastwam_num_frames),
        "--fastwam-action-video-freq-ratio", str(args.fastwam_action_video_freq_ratio),
        "--fastwam-global-sample-stride", str(args.fastwam_global_sample_stride),
        "--n-frames", str(args.n_frames),
        "--num-history", str(args.num_history),
        "--chunk-size", str(args.chunk_size),
        "--n-action-steps", str(args.n_action_steps),
        "--num-views", str(args.num_views),
        "--batch-size", str(candidate),
        "--num-workers", str(args.num_workers),
        "--max-steps", str(args.probe_steps),
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--warmup-steps", str(args.warmup_steps),
        "--grad-clip", str(args.grad_clip),
        "--grad-accum-steps", str(args.grad_accum_steps),
        "--log-every", str(args.log_every),
        "--save-every", str(args.save_every),
        "--seed", str(args.seed),
        "--device", args.device,
        "--dtype", args.dtype,
        "--max-text-length", str(args.max_text_length),
        "--probe-candidate-batch", str(candidate),
        "--no-save",
    ]
    if args.text_cache_path:
        child.extend(["--text-cache-path", args.text_cache_path])
    if args.resume:
        child.extend(["--resume", args.resume])
    if args.find_unused_parameters:
        child.append("--find-unused-parameters")
    return child


def _is_oom_output(text: str) -> bool:
    lower = text.lower()
    return any(marker in lower for marker in OOM_MARKERS)


def _tail(text: str, lines: int = 80) -> str:
    split = text.rstrip().splitlines()
    return "\n".join(split[-lines:])


def _run_probe_candidate(args, candidate: int, gpus: list[str]) -> bool:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
    world_size = len(gpus)
    cmd = [
        sys.executable,
        "-m", "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={world_size}",
        str(Path(__file__).resolve()),
        *_child_args(args, candidate),
    ]
    print(f"[probe] testing global_batch={candidate} per_rank_batch={candidate // world_size}", flush=True)
    result = subprocess.run(
        cmd,
        cwd=str(Path(__file__).resolve().parent),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output = result.stdout or ""
    if result.returncode == 0:
        for line in output.splitlines():
            if "[probe-rank]" in line or "[probe-pass]" in line:
                print(line, flush=True)
        return True
    if _is_oom_output(output):
        print(f"[probe-oom] global_batch={candidate}", flush=True)
        print(_tail(output, 40), flush=True)
        return False
    print(f"[probe-error] global_batch={candidate} returncode={result.returncode}", flush=True)
    print(_tail(output), flush=True)
    raise RuntimeError("probe child failed for a non-OOM reason")


def _round_up_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _round_down_multiple(value: int, multiple: int) -> int:
    return (value // multiple) * multiple


def probe_max_batch(args) -> None:
    if "LOCAL_RANK" in os.environ:
        raise RuntimeError("--probe-max-batch must be run as a single parent process, not under torchrun")
    if args.probe_steps < 1:
        raise ValueError("--probe-steps must be >= 1")
    gpus = _parse_probe_gpus(args.probe_gpus)
    world_size = len(gpus)
    start = args.probe_start_global_batch or world_size
    start = _round_up_multiple(max(start, world_size), world_size)
    max_batch = _round_down_multiple(args.probe_max_global_batch, world_size)
    if max_batch < start:
        raise ValueError(
            f"--probe-max-global-batch={args.probe_max_global_batch} is below start={start} "
            f"after rounding to world_size={world_size}"
        )

    print(
        f"[probe] gpus={','.join(gpus)} world_size={world_size} "
        f"start={start} max={max_batch} steps={args.probe_steps}",
        flush=True,
    )

    last_pass = 0
    first_fail = None
    candidate = start
    tested_max = False

    while candidate <= max_batch:
        ok = _run_probe_candidate(args, candidate, gpus)
        if ok:
            last_pass = candidate
            if candidate == max_batch:
                tested_max = True
                break
            candidate = min(candidate * 2, max_batch)
        else:
            first_fail = candidate
            break

    if first_fail is None and not tested_max and last_pass < max_batch:
        ok = _run_probe_candidate(args, max_batch, gpus)
        if ok:
            last_pass = max_batch
            tested_max = True
        else:
            first_fail = max_batch

    if first_fail is None:
        print(
            f"[probe-result] max global batch is at least {last_pass} "
            f"(per_rank_batch={last_pass // world_size}); increase --probe-max-global-batch to continue.",
            flush=True,
        )
        return

    low_units = last_pass // world_size
    high_units = first_fail // world_size - 1
    while low_units < high_units:
        mid_units = (low_units + high_units + 1) // 2
        candidate = mid_units * world_size
        ok = _run_probe_candidate(args, candidate, gpus)
        if ok:
            low_units = mid_units
            last_pass = candidate
        else:
            high_units = mid_units - 1

    result = low_units * world_size
    if result == 0:
        print("[probe-result] no tested global batch fit in memory", flush=True)
    else:
        print(
            f"[probe-result] max_global_batch={result} per_rank_batch={result // world_size} "
            f"first_oom_global_batch={first_fail}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    if args.probe_max_batch:
        probe_max_batch(args)
    else:
        train_main(args)


if __name__ == "__main__":
    main()
