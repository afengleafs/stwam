"""Ablation finetunes from a trained STWAM checkpoint (proprio dropout,
k-draws, pooled-adaLN connector).

Differences from ``train_ddp.py``:
  * ``--init-checkpoint`` loads *weights only* (fresh optimizer/scheduler,
    step reset to 0) — unlike ``--resume`` which restores the full run state;
  * the loss is ``model.ablation.training_loss_ablation`` with the run's
    ``--k-draws`` / ``--proprio-dropout`` factors;
  * ``--pooled-adaln {off,add,only}`` builds the pooled connector (new,
    zero-init params trained at ``--pooled-lr``); ``only`` additionally zeroes
    and freezes the adapters' layer-wise action-read path;
  * finetune defaults: lr 3e-5, warmup 200, 20k steps, save every 5k.

Example (R1, proprio dropout, 2-GPU DDP):
    CUDA_VISIBLE_DEVICES=6,7 .venv/bin/python -m torch.distributed.run \
      --standalone --nproc_per_node=2 --rdzv-endpoint=localhost:29511 train_ablation.py \
      --run-name r1_pdrop05 --proprio-dropout 0.5 --batch-size 32
"""
from __future__ import annotations

import argparse
import math
import random
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from model.ablation import freeze_layerwise_read_path, training_loss_ablation
from policy.stwam_policy import STWAMPolicy
from train import (
    _attach_text,
    _dtype,
    _move_batch,
    _require_imports,
    _task_dict_from_metadata,
    save_checkpoint,
)
from train_ddp import (
    _build_loader,
    _build_or_load_text_cache_ranked,
    _cleanup_distributed,
    _freeze_unused_train_path,
    _init_distributed,
    _rank0,
)
from train_libero import _adapt_fastwam_lerobot_batch, _make_config


class STWAMAblationPolicy(STWAMPolicy):
    """STWAMPolicy whose training forward runs the ablation loss."""

    k_draws: int = 8
    proprio_dropout: float = 0.0

    def forward(self, batch: dict):
        z = self._encode_video(batch)
        ctx, ctx_mask = self._context(batch)
        return training_loss_ablation(
            self.model, z, batch["action"], ctx, ctx_mask,
            k_draws=self.k_draws, proprio_dropout=self.proprio_dropout,
            action_is_pad=batch.get("action_is_pad"),
            image_is_pad=batch.get("image_is_pad"),
        )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True, help="checkpoint/ablation_<run-name>/")
    p.add_argument("--init-checkpoint", default="checkpoint/stwam_libero_ddp/latest.pt",
                   help="weights-only init (fresh optimizer, step 0)")
    # ablation factors
    p.add_argument("--k-draws", type=int, default=8)
    p.add_argument("--proprio-dropout", type=float, default=0.0)
    p.add_argument("--loss-lambda-video", type=float, default=1.0,
                   help="weight on the video-prediction loss; 0.0 = no video co-training "
                        "(world-model existence ablation, P0-1)")
    p.add_argument("--pooled-adaln", choices=("off", "add", "only"), default="off")
    p.add_argument("--pooled-queries", type=int, default=8)
    p.add_argument("--pooled-lr", type=float, default=1e-4,
                   help="lr for the (zero-init, from-scratch) pooled connector params")
    # data (identical to the original run)
    p.add_argument("--dataset-root", default="libero")
    p.add_argument("--video-dit-ckpt", default="weights/vjepa/DiT-S_D96.pt")
    p.add_argument("--adapter-ckpt", default="weights/vjepa/adapter_vjepa_image_96.pt")
    p.add_argument("--vjepa2-ckpt", default="weights/vjepa/vjepa2_1_vitl_dist_vitG_384.pt")
    p.add_argument("--text-model-id", default="google/flan-t5-large")
    p.add_argument("--text-model-dir", default="weights/flan_t5_large")
    p.add_argument("--text-cache-path", default=None)
    p.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    p.add_argument("--fastwam-num-frames", type=int, default=33)
    p.add_argument("--fastwam-action-video-freq-ratio", type=int, default=4)
    p.add_argument("--fastwam-global-sample-stride", type=int, default=1)
    p.add_argument("--n-frames", type=int, default=9)
    p.add_argument("--num-history", type=int, default=1)
    p.add_argument("--chunk-size", type=int, default=32)
    p.add_argument("--n-action-steps", type=int, default=32)
    p.add_argument("--num-views", type=int, default=2)
    # finetune schedule
    p.add_argument("--batch-size", type=int, default=32,
                   help="global batch; per-rank = batch_size / world_size")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max-text-length", type=int, default=128)
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--find-unused-parameters", action="store_true")
    return p.parse_args()


def _load_init_weights(policy: STWAMPolicy, path: str, pooled_on: bool, rank: int) -> None:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    missing, unexpected = policy.load_state_dict(ckpt["policy"], strict=False)
    ok_missing = [k for k in missing
                  if "vjepa" in k or "backbone" in k
                  or (pooled_on and ".pooled_cond." in k)]
    bad_missing = [k for k in missing if k not in ok_missing]
    if bad_missing or unexpected:
        raise RuntimeError(
            f"init checkpoint did not load cleanly: missing={bad_missing[:8]} "
            f"unexpected={list(unexpected)[:8]}")
    _rank0(rank, f"[init] {path} step={ckpt.get('step')} "
                 f"(fresh pooled params: {sum('.pooled_cond.' in k for k in missing)})")


def train_main(args) -> None:
    local_rank, rank, world_size, distributed = _init_distributed()
    try:
        _require_imports()
        if args.batch_size % world_size != 0:
            raise ValueError(f"--batch-size={args.batch_size} not divisible by world_size={world_size}")
        args.world_size = world_size
        args.distributed = distributed
        args.per_rank_batch = args.batch_size // world_size
        args.output_dir = f"checkpoint/ablation_{args.run_name}"
        args.resume = None  # _build_loader/_make_config compatibility

        random.seed(args.seed + rank)
        torch.manual_seed(args.seed + rank)
        if distributed:
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
        dtype = _dtype(args.dtype)

        from model import checkpoint as ck
        sd = ck.load_raw_state_dict(args.video_dit_ckpt)
        info = ck.introspect(sd)
        (dataset_metadata, dataset, sampler, loader, delta_timestamps,
         video_indices, state_indices, action_indices) = _build_loader(
            args, rank, world_size, distributed, device)
        cfg = _make_config(args, info, video_frame_count=len(video_indices),
                           action_count=len(action_indices))
        cfg.pooled_adaln = args.pooled_adaln
        cfg.pooled_queries = args.pooled_queries
        cfg.loss_lambda_video = args.loss_lambda_video

        tasks = _task_dict_from_metadata(dataset_metadata)
        _rank0(rank, f"[ablation:{args.run_name}] k_draws={args.k_draws} "
                     f"proprio_dropout={args.proprio_dropout} pooled_adaln={args.pooled_adaln} "
                     f"loss_lambda_video={cfg.loss_lambda_video}")
        _rank0(rank, f"[ddp] world_size={world_size} global_batch={args.batch_size} "
                     f"per_rank_batch={args.per_rank_batch} accum={args.grad_accum_steps}")
        _rank0(rank, f"[data] frames={len(dataset)} tasks={len(tasks)} fps={dataset_metadata.fps}")
        text_embeds, text_mask = _build_or_load_text_cache_ranked(args, tasks, device, dtype, rank, distributed)

        policy = STWAMAblationPolicy(cfg).to(device)
        policy.k_draws = args.k_draws
        policy.proprio_dropout = args.proprio_dropout
        _load_init_weights(policy, args.init_checkpoint, args.pooled_adaln != "off", rank)
        _freeze_unused_train_path(policy)
        if args.pooled_adaln == "only":
            frozen = freeze_layerwise_read_path(policy.model)
            _rank0(rank, f"[pooled-only] zeroed+froze {len(frozen)} read-path entries "
                         f"across {len(policy.model.adapters)} adapters")

        if distributed:
            policy = DistributedDataParallel(
                policy, device_ids=[local_rank], output_device=local_rank,
                broadcast_buffers=False, find_unused_parameters=args.find_unused_parameters)
            policy_no_ddp = policy.module
        else:
            policy_no_ddp = policy

        pooled_params, base_params = [], []
        for name, prm in policy_no_ddp.named_parameters():
            if not prm.requires_grad:
                continue
            (pooled_params if ".pooled_cond." in name else base_params).append(prm)
        groups = [{"params": base_params, "lr": args.lr}]
        if pooled_params:
            groups.append({"params": pooled_params, "lr": args.pooled_lr})
        optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
        _rank0(rank, f"[optim] base params={sum(p.numel() for p in base_params)/1e6:.1f}M @lr={args.lr} "
                     f"pooled params={sum(p.numel() for p in pooled_params)/1e6:.2f}M @lr={args.pooled_lr}")

        def lr_lambda(step: int) -> float:
            if step < args.warmup_steps:
                return max(step, 1) / max(args.warmup_steps, 1)
            progress = (step - args.warmup_steps) / max(args.max_steps - args.warmup_steps, 1)
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        policy.train()
        out_dir = Path(args.output_dir)
        step = 0
        epoch = 0
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(total=args.max_steps, desc=f"ablation:{args.run_name}", disable=rank != 0)

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
                        f"video={parts['loss_video']:.4f} action={parts['loss_action']:.4f} lr={lr:.2e}")
                if rank == 0 and not args.no_save and step % args.save_every == 0:
                    save_checkpoint(out_dir / f"step_{step:08d}.pt", policy_no_ddp, optimizer, scheduler, step, cfg, args)
            epoch += 1

        if rank == 0:
            pbar.close()
            if not args.no_save:
                save_checkpoint(out_dir / "latest.pt", policy_no_ddp, optimizer, scheduler, step, cfg, args)
                print(f"[done] saved {out_dir / 'latest.pt'}", flush=True)
    finally:
        _cleanup_distributed(distributed)


if __name__ == "__main__":
    train_main(parse_args())
