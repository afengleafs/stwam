#!/usr/bin/env bash
# C3 P-only connector: 300k from-scratch on the last 6 GPUs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

NPROC="$(awk -F',' '{print NF}' <<<"${CUDA_VISIBLE_DEVICES}")"
RUN_NAME="${RUN_NAME:-stwam_pooled_only}"
OUT_DIR="${OUT_DIR:-checkpoint/${RUN_NAME}}"
mkdir -p logs "${OUT_DIR}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-logs/${RUN_NAME}_${STAMP}.log}"
PID_FILE="${PID_FILE:-logs/${RUN_NAME}.pid}"

# Global batch must be divisible by nproc. 30 ≈ main run's 32; per-rank = 5.
nohup .venv/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node="${NPROC}" train_ddp.py \
  --dataset-root libero \
  --connector-mode pooled_only \
  --pooled-queries 8 \
  --batch-size 30 \
  --grad-accum-steps 1 \
  --num-workers 0 \
  --max-steps 300000 \
  --save-every 100000 \
  --output-dir "${OUT_DIR}" \
  > "${LOG_FILE}" 2>&1 &

PID="$!"
echo "${PID}" > "${PID_FILE}"

echo "Started P-only (pooled_only) 300k training in background."
echo "PID:       ${PID}"
echo "GPUs:      ${CUDA_VISIBLE_DEVICES} (nproc=${NPROC})"
echo "Log:       ${ROOT}/${LOG_FILE}"
echo "PID file:  ${ROOT}/${PID_FILE}"
echo "Ckpts:     ${ROOT}/${OUT_DIR}/"
echo "Plan:      /mnt/sdb/feng/C3_Connector_Plan.md"
echo ""
echo "Monitor:"
echo "  tail -f ${ROOT}/${LOG_FILE}"
echo "  ls -lh ${ROOT}/${OUT_DIR}/"
