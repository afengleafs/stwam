#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

RUN_NAME="${RUN_NAME:-vtwam_libero_ddp}"
OUTPUT_DIR="${OUTPUT_DIR:-vtwam/checkpoint/${RUN_NAME}}"
LOG_FILE="${LOG_FILE:-logs/vtwam_libero_ddp_$(date +%Y%m%d_%H%M%S).log}"
PID_FILE="${PID_FILE:-logs/vtwam_libero_ddp.pid}"

mkdir -p logs "${OUTPUT_DIR}"

nohup bash scripts/train_vtwam_libero_ddp.sh \
  > "${LOG_FILE}" 2>&1 &

PID="$!"
echo "${PID}" > "${PID_FILE}"

echo "Started VTWAM LIBERO DDP training in background."
echo "PID: ${PID}"
echo "Log: ${LOG_FILE}"
echo "PID file: ${PID_FILE}"
