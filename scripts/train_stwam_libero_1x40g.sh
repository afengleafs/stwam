#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

RUN_NAME="${RUN_NAME:-stwam_libero_1x40g_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoint/${RUN_NAME}}"

mkdir -p "${OUTPUT_DIR}" logs

exec .venv/bin/python train.py \
  --dataset-root libero \
  --video-dit-ckpt weights/vjepa/DiT-S_D96.pt \
  --adapter-ckpt weights/vjepa/adapter_vjepa_image_96.pt \
  --vjepa2-ckpt weights/vjepa/vjepa2_1_vitl_dist_vitG_384.pt \
  --text-model-id google/flan-t5-large \
  --text-model-dir weights/flan_t5_large \
  --text-cache-path weights/flan_t5_large/libero_text_cache.pt \
  --hf-endpoint https://hf-mirror.com \
  --output-dir "${OUTPUT_DIR}" \
  --num-views 2 \
  --n-frames 8 \
  --num-history 2 \
  --chunk-size 16 \
  --n-action-steps 8 \
  --batch-size "${BATCH_SIZE:-1}" \
  --num-workers "${NUM_WORKERS:-2}" \
  --max-steps "${MAX_STEPS:-300000}" \
  --lr "${LR:-5e-5}" \
  --weight-decay 0.01 \
  --warmup-steps "${WARMUP_STEPS:-1000}" \
  --grad-clip 1.0 \
  --grad-accum-steps "${GRAD_ACCUM_STEPS:-1}" \
  --log-every "${LOG_EVERY:-10}" \
  --save-every "${SAVE_EVERY:-5000}" \
  --device cuda \
  --dtype bfloat16
