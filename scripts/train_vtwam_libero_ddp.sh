#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
RUN_NAME="${RUN_NAME:-vtwam_libero_ddp}"
OUTPUT_DIR="${OUTPUT_DIR:-vtwam/checkpoint/${RUN_NAME}}"

mkdir -p "${OUTPUT_DIR}" logs

exec .venv/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  --module vtwam.train_ddp \
  --dataset-root "${DATASET_ROOT:-libero}" \
  --video-dit-ckpt "${VIDEO_DIT_CKPT:-vtwam/checkpoint/vae/DiT-S_D16.pt}" \
  --vae-model-dir "${VAE_MODEL_DIR:-vtwam/checkpoint/sd3-medium-diffusers}" \
  --text-model-id "${TEXT_MODEL_ID:-google/flan-t5-large}" \
  --text-model-dir "${TEXT_MODEL_DIR:-weights/flan_t5_large}" \
  --text-cache-path "${TEXT_CACHE_PATH:-weights/flan_t5_large/libero_text_cache.pt}" \
  --hf-endpoint "${HF_ENDPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --fastwam-num-frames "${FASTWAM_NUM_FRAMES:-33}" \
  --fastwam-action-video-freq-ratio "${FASTWAM_ACTION_VIDEO_FREQ_RATIO:-4}" \
  --fastwam-global-sample-stride "${FASTWAM_GLOBAL_SAMPLE_STRIDE:-1}" \
  --num-views "${NUM_VIEWS:-2}" \
  --n-frames "${N_FRAMES:-9}" \
  --num-history "${NUM_HISTORY:-1}" \
  --chunk-size "${CHUNK_SIZE:-32}" \
  --n-action-steps "${N_ACTION_STEPS:-32}" \
  --batch-size "${BATCH_SIZE:-32}" \
  --num-workers "${NUM_WORKERS:-0}" \
  --max-steps "${MAX_STEPS:-300000}" \
  --lr "${LR:-1e-4}" \
  --weight-decay "${WEIGHT_DECAY:-0.01}" \
  --warmup-steps "${WARMUP_STEPS:-500}" \
  --grad-clip "${GRAD_CLIP:-1.0}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS:-1}" \
  --log-every "${LOG_EVERY:-10}" \
  --save-every "${SAVE_EVERY:-100000}" \
  --device cuda \
  --dtype "${DTYPE:-bfloat16}" \
  "$@"
