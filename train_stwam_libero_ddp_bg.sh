#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

mkdir -p logs checkpoint/stwam_libero_ddp

LOG_FILE="${LOG_FILE:-logs/stwam_libero_ddp_$(date +%Y%m%d_%H%M%S).log}"
PID_FILE="${PID_FILE:-logs/stwam_libero_ddp.pid}"

nohup .venv/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=4 train_ddp.py \
  --dataset-root libero \
  --batch-size 32 \
  --grad-accum-steps 1 \
  --num-workers 0 \
  --output-dir checkpoint/stwam_libero_ddp \
  --save-every 100000 \
  > "${LOG_FILE}" 2>&1 &

PID="$!"
echo "${PID}" > "${PID_FILE}"

echo "Started STWAM LIBERO DDP training in background."
echo "PID: ${PID}"
echo "Log: ${LOG_FILE}"
echo "PID file: ${PID_FILE}"
