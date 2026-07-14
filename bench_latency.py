"""P0-4: STWAM action-inference latency benchmark.

Isolates the WAM tower latency (V-JEPA encoder is not built here; the paper's
single-prefill claim is about the video+action towers). Compares:

  A) prefill-once + N flow steps  -> the deployed ``sample_actions`` path;
  B) naive: re-run the video-tower prefill inside *every* flow step.

Reports mean/std ms per action chunk, amortized ms per executed action, the
equivalent control frequency, and the A-vs-B speedup that the cached per-layer
K/V buys.

Run:
  CUDA_VISIBLE_DEVICES=4 .venv/bin/python bench_latency.py \
      --checkpoint checkpoint/stwam_libero_ddp/latest.pt --device cuda:0 --iters 100
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch

from model.config import STWAMConfig
from policy.stwam_policy import STWAMPolicy


def load_wam_only(checkpoint: str, device: torch.device):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = STWAMConfig(**dict(ckpt["config"]))
    cfg.device = str(device)
    cfg.adapter_ckpt = None  # do NOT build the V-JEPA encoder; time WAM towers only
    policy = STWAMPolicy(cfg).to(device)
    missing, unexpected = policy.load_state_dict(ckpt["policy"], strict=False)
    bad = [k for k in missing if "vjepa" not in k and "backbone" not in k and "adapter" not in k]
    if bad:
        print(f"[warn] {len(bad)} unexpected missing WAM keys: {bad[:6]}")
    policy.eval()
    print(f"[policy] step={ckpt.get('step')} chunk={cfg.chunk_size} n_action_steps={cfg.n_action_steps} "
          f"in_ch={cfg.in_channels} num_views={cfg.num_views} flow_steps={cfg.action_sampling_steps}")
    return policy, cfg


@torch.no_grad()
def naive_sample_actions(model, anchor_lat, ctx, ctx_mask, num_steps):
    """Path B: re-prefill the video tower at every flow step (no K/V cache reuse)."""
    B = anchor_lat.shape[0]
    device = anchor_lat.device
    a_ctx = model.action.embed_context(ctx)
    x_a = torch.randn(B, model.config.chunk_size, model.config.action_dim, device=device)
    schedule = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    for i in range(num_steps):
        t_cur, t_next = schedule[i], schedule[i + 1]
        dt = t_cur - t_next
        cache = model.prefill_video(anchor_lat, ctx, ctx_mask)  # <-- re-run every step
        ca = model.action.cond(torch.full((B,), float(t_cur), device=device))
        xa = model.action.embed(x_a)
        for L in range(model.num_layers):
            xa = model.action.blocks[L](xa, ca, a_ctx, ctx_mask)
            xa = model.adapters[L].update_action(xa, *cache[L], model.action_times)
        v_a = model.action.head_out(xa)
        x_a = x_a - dt * v_a
    return x_a


def timeit(fn, iters, device, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize(device)
        ts.append((time.perf_counter() - t0) * 1000.0)
    return statistics.mean(ts), statistics.pstdev(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoint/stwam_libero_ddp/latest.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--output", default="logs/latency/latency.txt")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
             "float16": torch.float16, "float32": torch.float32}[args.dtype]

    policy, cfg = load_wam_only(args.checkpoint, device)
    model = policy.model

    B = 1
    Hs, Ws = 16, 16 * cfg.num_views
    anchor = torch.randn(B, cfg.num_history, Hs, Ws, cfg.in_channels, device=device)
    Ltxt = 20
    text = torch.randn(B, Ltxt, cfg.text_dim, device=device)
    text_mask = torch.ones(B, Ltxt, dtype=torch.bool, device=device)
    proprio = torch.randn(B, cfg.proprio_dim, device=device)
    ctx, ctx_mask = model.build_context(text, text_mask, proprio)

    def pathA():
        with torch.autocast(device_type=device.type, dtype=dtype):
            model.sample_actions(anchor, ctx, ctx_mask)

    def pathB():
        with torch.autocast(device_type=device.type, dtype=dtype):
            naive_sample_actions(model, anchor, ctx, ctx_mask, cfg.action_sampling_steps)

    a_mean, a_std = timeit(pathA, args.iters, device)
    b_mean, b_std = timeit(pathB, args.iters, device)

    per_exec = a_mean / max(cfg.n_action_steps, 1)
    hz_chunk = 1000.0 / a_mean
    hz_amort = 1000.0 / per_exec
    gpu = torch.cuda.get_device_name(device)

    lines = [
        "STWAM action-inference latency (P0-4)",
        f"device={gpu}  dtype={args.dtype}  iters={args.iters}  batch={B}",
        f"flow_steps={cfg.action_sampling_steps}  chunk_size={cfg.chunk_size}  "
        f"n_action_steps(replan)={cfg.n_action_steps}",
        "",
        f"A) prefill-once (deployed):     {a_mean:7.2f} ± {a_std:5.2f} ms / chunk",
        f"B) naive re-prefill per step:   {b_mean:7.2f} ± {b_std:5.2f} ms / chunk",
        f"   single-prefill speedup A vs B: {b_mean / a_mean:5.2f}x",
        "",
        f"amortized planning / executed action = {per_exec:6.2f} ms  "
        f"(replan every {cfg.n_action_steps} actions)",
        f"planning throughput: {hz_chunk:6.1f} chunks/s  |  {hz_amort:7.1f} action-plans/s (amortized)",
        "",
        "note: V-JEPA+S-VAE encode cost is excluded (once per control step, measured separately).",
    ]
    report = "\n".join(lines)
    print("\n" + report)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(report + "\n")
    print(f"\n[done] wrote {args.output}")


if __name__ == "__main__":
    main()
