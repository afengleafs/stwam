"""Train STWAM on local LIBERO with FastWAM-style timestamps.

FastWAM's LIBERO window uses 33 raw 10 Hz steps:
  - actions at steps 0..31        -> 32 actions
  - video frames at 0,4,..,32     -> 9 frames
  - state at steps 0..32          -> 33 states

This script is the single-process training entrypoint.  The DDP entrypoint
remains ``train_ddp.py``.
"""
from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import checkpoint as ck
from model.config import STWAMConfig
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


def build_fastwam_libero_delta_timestamps(
    fps: float,
    num_frames: int = 33,
    action_video_freq_ratio: int = 4,
    global_sample_stride: int = 1,
) -> tuple[dict[str, list[float]], list[int], list[int], list[int]]:
    """Return LeRobot timestamps matching FastWAM's LIBERO data window."""
    if num_frames < 2:
        raise ValueError("--fastwam-num-frames must be >= 2")
    if action_video_freq_ratio < 1:
        raise ValueError("--fastwam-action-video-freq-ratio must be >= 1")
    if global_sample_stride < 1:
        raise ValueError("--fastwam-global-sample-stride must be >= 1")
    if (num_frames - 1) % action_video_freq_ratio != 0:
        raise ValueError(
            f"fastwam_num_frames - 1 must be divisible by "
            f"fastwam_action_video_freq_ratio, got {num_frames - 1} and "
            f"{action_video_freq_ratio}"
        )

    state_indices = list(range(num_frames))
    action_indices = list(range(num_frames - 1))
    video_indices = list(range(0, num_frames, action_video_freq_ratio))

    def to_seconds(indices: list[int]) -> list[float]:
        return [(i * global_sample_stride) / fps for i in indices]

    delta_timestamps = {
        "observation.images.image": to_seconds(video_indices),
        "observation.images.image2": to_seconds(video_indices),
        "observation.state": to_seconds(state_indices),
        "action": to_seconds(action_indices),
    }
    return delta_timestamps, video_indices, state_indices, action_indices


def _make_config(args, info: dict[str, Any], video_frame_count: int, action_count: int) -> STWAMConfig:
    if args.n_frames != video_frame_count:
        raise ValueError(
            f"--n-frames={args.n_frames} must match FastWAM video frame count "
            f"{video_frame_count}; adjust --fastwam-num-frames or "
            f"--fastwam-action-video-freq-ratio instead"
        )
    if args.chunk_size != action_count:
        raise ValueError(
            f"--chunk-size={args.chunk_size} must match FastWAM action count "
            f"{action_count}; adjust --fastwam-num-frames instead"
        )
    if args.num_history < 1 or args.num_history > args.n_frames:
        raise ValueError("--num-history must be in [1, n_frames]")
    if args.n_action_steps > args.chunk_size:
        raise ValueError("--n-action-steps cannot be greater than --chunk-size")

    cfg = STWAMConfig(
        video_dit_ckpt=args.video_dit_ckpt,
        adapter_ckpt=args.adapter_ckpt,
        vjepa2_ckpt=args.vjepa2_ckpt,
        n_frames=args.n_frames,
        num_history=args.num_history,
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        frame_skip=args.fastwam_global_sample_stride,
        num_views=args.num_views,
        observation_delta_indices=None,
        action_delta_indices=None,
        device=args.device,
        dtype=args.dtype,
        text_encoder_name=args.text_model_id,
        proprio_dim=8,
        max_state_dim=8,
    )
    cfg.apply_introspection(info)
    cfg.objective, cfg.temporal_mode = "ddpm", "factored"
    cfg.action_layers = cfg.num_layers
    return cfg


def _adapt_fastwam_lerobot_batch(batch: dict[str, Any]) -> dict[str, Any]:
    """Convert FastWAM-window LeRobot output to STWAMPolicy batch keys."""
    out = dict(batch)
    state = out.get("observation.state")
    if torch.is_tensor(state) and state.ndim == 3:
        out["observation.state"] = state[:, 0]

    pad_masks = [
        out[k].bool()
        for k in ("observation.images.image_is_pad", "observation.images.image2_is_pad")
        if k in out
    ]
    if pad_masks:
        image_is_pad = pad_masks[0]
        for pad in pad_masks[1:]:
            image_is_pad = image_is_pad | pad
        out["image_is_pad"] = image_is_pad
    return out


def _freeze_unused_train_path(policy: STWAMPolicy) -> None:
    """Freeze semantic-wm DiT action conditioning, unused by STWAM's path."""
    action_embedder = getattr(policy.model.video.dit, "action_embedder", None)
    if action_embedder is not None:
        action_embedder.requires_grad_(False)


def _build_loader(args, device: torch.device):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    dataset_metadata = LeRobotDatasetMetadata("local/libero", root=args.dataset_root)
    delta_timestamps, video_indices, state_indices, action_indices = build_fastwam_libero_delta_timestamps(
        dataset_metadata.fps,
        num_frames=args.fastwam_num_frames,
        action_video_freq_ratio=args.fastwam_action_video_freq_ratio,
        global_sample_stride=args.fastwam_global_sample_stride,
    )
    dataset = LeRobotDataset(
        "local/libero",
        root=args.dataset_root,
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    return dataset_metadata, dataset, loader, delta_timestamps, video_indices, state_indices, action_indices


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

    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=300000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=100000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max-text-length", type=int, default=128)
    p.add_argument("--no-save", action="store_true")
    return p.parse_args()


def main() -> None:
    _require_imports()
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.grad_accum_steps <= 0:
        raise ValueError("--grad-accum-steps must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = _dtype(args.dtype)

    dataset_metadata, dataset, loader, delta_timestamps, video_indices, state_indices, action_indices = _build_loader(
        args, device
    )
    print(f"[data] frames={len(dataset)} tasks={len(_task_dict_from_metadata(dataset_metadata))} fps={dataset_metadata.fps}")
    print(f"[data] video_indices={video_indices}")
    print(f"[data] state_indices={state_indices[0]}..{state_indices[-1]} len={len(state_indices)}")
    print(f"[data] action_indices={action_indices[0]}..{action_indices[-1]} len={len(action_indices)}")
    print(f"[data] delta_timestamps={delta_timestamps}")

    if args.max_steps == 0:
        print("[done] max_steps=0; validated dataset timestamps only")
        return

    for path in (args.video_dit_ckpt, args.adapter_ckpt, args.vjepa2_ckpt):
        if not Path(path).is_file():
            raise FileNotFoundError(path)

    sd = ck.load_raw_state_dict(args.video_dit_ckpt)
    info = ck.introspect(sd)
    cfg = _make_config(args, info, video_frame_count=len(video_indices), action_count=len(action_indices))

    tasks = _task_dict_from_metadata(dataset_metadata)
    text_embeds, text_mask = _build_text_cache(args, tasks, device, dtype)
    print(f"[text] cache embeds={tuple(text_embeds.shape)} mask={tuple(text_mask.shape)}")

    policy = STWAMPolicy(cfg).to(device)
    missing, unexpected = policy.model.video.load_pretrained(sd, strict=False)
    print(f"[video DiT] load: {len(missing)} missing / {len(unexpected)} unexpected")
    if missing or unexpected:
        raise RuntimeError("video DiT checkpoint did not load cleanly")
    _freeze_unused_train_path(policy)

    optim_params = [p for p in policy.get_optim_params() if p.requires_grad]
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
        policy.load_state_dict(ckpt["policy"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_step = int(ckpt.get("step", 0))
        print(f"[resume] {args.resume} step={start_step}")

    policy.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    out_dir = Path(args.output_dir)
    step = start_step
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(total=args.max_steps, initial=step, desc="train")

    while step < args.max_steps:
        for batch in loader:
            if step >= args.max_steps:
                break
            batch = _adapt_fastwam_lerobot_batch(batch)
            batch = _attach_text(batch, text_embeds, text_mask)
            batch = _move_batch(batch, device)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                loss, parts = policy(batch)
                loss = loss / args.grad_accum_steps
            loss.backward()
            if (step + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            step += 1
            pbar.update(1)
            if step % args.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                pbar.write(
                    f"step={step} loss={float(loss.detach()) * args.grad_accum_steps:.4f} "
                    f"video={parts['loss_video']:.4f} action={parts['loss_action']:.4f} lr={lr:.2e}"
                )
            if not args.no_save and step % args.save_every == 0:
                save_checkpoint(out_dir / f"step_{step:08d}.pt", policy, optimizer, scheduler, step, cfg, args)

    pbar.close()
    if not args.no_save:
        save_checkpoint(out_dir / "latest.pt", policy, optimizer, scheduler, step, cfg, args)
        print(f"[done] saved {out_dir / 'latest.pt'}")


if __name__ == "__main__":
    main()
