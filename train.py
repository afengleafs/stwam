"""Train STWAM on the local LIBERO LeRobot-v3 dataset.

The script intentionally keeps the training loop close to LeRobot/pi0.5's
shape: build a dataset, assemble a policy batch, call ``policy(batch)`` for
``(loss, parts)``, then run the optimizer step.  Dataset loading follows the
official LeRobot example: build ``LeRobotDatasetMetadata``, define
``delta_timestamps`` from policy-style delta indices, then instantiate
``LeRobotDataset`` with those timestamps.

Run a smoke test:
    .venv/bin/python train.py --max-steps 2 --batch-size 1 --num-workers 0
"""
from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import checkpoint as ck
from model.config import STWAMConfig
from policy.stwam_policy import STWAMPolicy


ALIYUN_UV_INSTALL = (
    "uv pip install -i https://mirrors.aliyun.com/pypi/simple/ "
    "--trusted-host mirrors.aliyun.com lerobot pyarrow av sentencepiece"
)


def _require_imports() -> None:
    missing = []
    for name in ("av", "pyarrow", "transformers", "huggingface_hub"):
        try:
            __import__(name)
        except Exception:
            missing.append(name)
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata  # noqa: F401
    except Exception:
        missing.append("lerobot")
    try:
        __import__("sentencepiece")
    except Exception:
        missing.append("sentencepiece")
    if missing:
        raise SystemExit(
            "Missing dependencies: "
            + ", ".join(sorted(set(missing)))
            + f"\nInstall with Aliyun mirror:\n  {ALIYUN_UV_INSTALL}"
        )


def _dtype(name: str) -> torch.dtype:
    table = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
             "fp16": torch.float16, "float16": torch.float16,
             "fp32": torch.float32, "float32": torch.float32}
    if name not in table:
        raise ValueError(f"unknown dtype {name!r}")
    return table[name]


def _default_observation_delta_indices(n_frames: int, num_history: int, frame_skip: int) -> tuple[int, ...]:
    return tuple((i - (num_history - 1)) * frame_skip for i in range(n_frames))


def _default_action_delta_indices(chunk_size: int) -> tuple[int, ...]:
    return tuple(range(chunk_size))


def build_delta_timestamps(cfg: STWAMConfig, fps: float) -> dict[str, list[float]]:
    obs_idx = cfg.observation_delta_indices
    if obs_idx is None:
        obs_idx = _default_observation_delta_indices(cfg.n_frames, cfg.num_history, cfg.frame_skip)
    act_idx = cfg.action_delta_indices
    if act_idx is None:
        act_idx = _default_action_delta_indices(cfg.chunk_size)
    return {
        "observation.images.image": [i / fps for i in obs_idx],
        "observation.images.image2": [i / fps for i in obs_idx],
        "observation.state": [i / fps for i in obs_idx],
        "action": [i / fps for i in act_idx],
    }


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def _task_dict_from_metadata(metadata) -> dict[int, str]:
    tasks = metadata.tasks
    if hasattr(tasks, "index") and hasattr(tasks, "__len__"):
        if hasattr(tasks, "columns") and "task_index" in tasks.columns:
            return {
                int(task_idx): str(task)
                for task_idx, task in zip(tasks["task_index"].tolist(), tasks.index.tolist())
            }
        return {int(i): str(task) for i, task in enumerate(tasks.index.tolist())}
    if isinstance(tasks, dict):
        return {int(k): str(v) for k, v in tasks.items()}
    raise TypeError(f"Unsupported LeRobot tasks metadata type: {type(tasks)}")


def _text_cache_path(args) -> Path:
    if args.text_cache_path:
        return Path(args.text_cache_path)
    return Path(args.text_model_dir) / "libero_text_cache.pt"


def _build_text_cache(args, tasks: dict[int, str], device: torch.device, dtype: torch.dtype):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, T5EncoderModel

    task_indices = sorted(tasks)
    if not task_indices:
        raise ValueError("LeRobot metadata has no tasks; cannot build text cache")
    cache_path = _text_cache_path(args)
    cache_meta = {
        "text_model_id": args.text_model_id,
        "max_text_length": args.max_text_length,
        "tasks": {int(i): tasks[i] for i in task_indices},
    }
    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("meta") == cache_meta:
            return payload["text_embeds"], payload["text_mask"]
        print(f"[text] ignoring stale cache: {cache_path}")

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    text_dir = Path(args.text_model_dir)
    if not text_dir.exists() or not any(text_dir.iterdir()):
        text_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=args.text_model_id,
            local_dir=str(text_dir),
            local_dir_use_symlinks=False,
            endpoint=args.hf_endpoint or None,
        )
    tokenizer = AutoTokenizer.from_pretrained(str(text_dir), local_files_only=True)
    encoder = T5EncoderModel.from_pretrained(str(text_dir), torch_dtype=dtype, local_files_only=True)
    encoder.eval().to(device)
    for p in encoder.parameters():
        p.requires_grad_(False)

    ordered = [tasks[i] for i in task_indices]
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
        toks = tokenizer(
            ordered,
            padding=True,
            truncation=True,
            max_length=args.max_text_length,
            return_tensors="pt",
        ).to(device)
        hidden = encoder(**toks).last_hidden_state.float().cpu()
        mask = toks.attention_mask.bool().cpu()
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Build a direct task_index -> cache row table.  Current LIBERO metadata is
    # contiguous 0..39, but this keeps the training path correct for sparse ids.
    rows = torch.tensor(task_indices, dtype=torch.long)
    hidden_table = hidden.new_zeros((int(rows.max()) + 1, *hidden.shape[1:]))
    mask_table = mask.new_zeros((int(rows.max()) + 1, *mask.shape[1:]))
    hidden_table[rows] = hidden
    mask_table[rows] = mask
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"meta": cache_meta, "text_embeds": hidden_table, "text_mask": mask_table}, cache_path)
    print(f"[text] saved cache: {cache_path}")
    return hidden_table, mask_table


def _attach_text(batch: dict[str, Any], text_embeds: torch.Tensor, text_mask: torch.Tensor) -> dict[str, Any]:
    idx = batch["task_index"].long().cpu()
    if idx.numel() > 0 and (idx.min() < 0 or idx.max() >= text_embeds.shape[0]):
        raise KeyError(f"batch contains task_index outside text cache: {idx.tolist()}")
    batch["text_embeds"] = text_embeds[idx]
    batch["text_mask"] = text_mask[idx]
    return batch


def _adapt_lerobot_batch(batch: dict[str, Any], cfg: STWAMConfig) -> dict[str, Any]:
    """Convert LeRobotDataset output to STWAMPolicy's expected batch keys."""
    out = dict(batch)
    state = out.get("observation.state")
    if torch.is_tensor(state) and state.ndim == 3:
        obs_idx = cfg.observation_delta_indices
        if obs_idx is None:
            obs_idx = _default_observation_delta_indices(cfg.n_frames, cfg.num_history, cfg.frame_skip)
        try:
            current_pos = list(obs_idx).index(0)
        except ValueError:
            current_pos = len(obs_idx) - 1
        out["observation.state"] = state[:, current_pos]
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


def _make_config(args, info: dict[str, Any]) -> STWAMConfig:
    obs_idx = tuple(args.observation_delta_indices) if args.observation_delta_indices else None
    act_idx = tuple(args.action_delta_indices) if args.action_delta_indices else None
    cfg = STWAMConfig(
        video_dit_ckpt=args.video_dit_ckpt,
        adapter_ckpt=args.adapter_ckpt,
        vjepa2_ckpt=args.vjepa2_ckpt,
        n_frames=args.n_frames,
        num_history=args.num_history,
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        frame_skip=args.frame_skip,
        num_views=args.num_views,
        observation_delta_indices=obs_idx,
        action_delta_indices=act_idx,
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


def save_checkpoint(path: Path, policy: STWAMPolicy, optimizer, scheduler, step: int,
                    cfg: STWAMConfig, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "policy": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "step": step,
        "config": asdict(cfg),
        "args": vars(args),
    }, path)


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
    p.add_argument("--output-dir", default="checkpoint/stwam_libero")
    p.add_argument("--resume", default=None)
    p.add_argument("--n-frames", type=int, default=8)
    p.add_argument("--num-history", type=int, default=2)
    p.add_argument("--chunk-size", type=int, default=16)
    p.add_argument("--n-action-steps", type=int, default=8)
    p.add_argument("--frame-skip", type=int, default=1)
    p.add_argument("--observation-delta-indices", type=int, nargs="*", default=None)
    p.add_argument("--action-delta-indices", type=int, nargs="*", default=None)
    p.add_argument("--num-views", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=10000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max-text-length", type=int, default=128)
    return p.parse_args()


def main() -> None:
    _require_imports()
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = _dtype(args.dtype)

    for path in (args.video_dit_ckpt, args.adapter_ckpt, args.vjepa2_ckpt):
        if not Path(path).is_file():
            raise FileNotFoundError(path)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    sd = ck.load_raw_state_dict(args.video_dit_ckpt)
    info = ck.introspect(sd)
    cfg = _make_config(args, info)
    dataset_metadata = LeRobotDatasetMetadata("local/libero", root=args.dataset_root)
    delta_timestamps = build_delta_timestamps(cfg, dataset_metadata.fps)
    dataset = LeRobotDataset(
        "local/libero", root=args.dataset_root, delta_timestamps=delta_timestamps,
        video_backend="pyav",
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=device.type == "cuda", drop_last=True, persistent_workers=args.num_workers > 0,
    )

    tasks = _task_dict_from_metadata(dataset_metadata)
    print(f"[data] frames={len(dataset)} tasks={len(tasks)} fps={dataset_metadata.fps}")
    print(f"[data] delta_timestamps={delta_timestamps}")
    text_embeds, text_mask = _build_text_cache(args, tasks, device, dtype)
    print(f"[text] cache embeds={tuple(text_embeds.shape)} mask={tuple(text_mask.shape)}")

    policy = STWAMPolicy(cfg).to(device)
    missing, unexpected = policy.model.video.load_pretrained(sd, strict=False)
    print(f"[video DiT] load: {len(missing)} missing / {len(unexpected)} unexpected")
    if missing or unexpected:
        raise RuntimeError("video DiT checkpoint did not load cleanly")

    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=args.lr, weight_decay=args.weight_decay)

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
    out_dir = Path(args.output_dir)
    step = start_step
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(total=args.max_steps, initial=step, desc="train")
    while step < args.max_steps:
        for batch in loader:
            if step >= args.max_steps:
                break
            batch = _adapt_lerobot_batch(batch, cfg)
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
            if step % args.save_every == 0:
                save_checkpoint(out_dir / f"step_{step:08d}.pt", policy, optimizer, scheduler, step, cfg, args)
    save_checkpoint(out_dir / "latest.pt", policy, optimizer, scheduler, step, cfg, args)
    pbar.close()
    print(f"[done] saved {out_dir / 'latest.pt'}")


if __name__ == "__main__":
    main()
