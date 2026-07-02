#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

CHECKPOINT="${CHECKPOINT:-checkpoint/stwam_libero_ddp/latest.pt}"
DEVICE="${DEVICE:-cuda:1}"
N_EP="${N_EPISODES:-10}"
IFS=',' read -r -a PERTS <<< "${PERTURBATIONS:-object,swap,language,task,environment}"

SUITES=(libero_spatial libero_object libero_goal libero_10)

mkdir -p eval_libero_plus/logs

for suite in "${SUITES[@]}"; do
  for pert in "${PERTS[@]}"; do
    pert="${pert//[[:space:]]/}"
    [[ -n "$pert" ]] || continue
    out="eval_libero_plus/logs/eval_pro_${suite}_${pert}.csv"
    if [[ -f "$out" ]]; then
      echo "[skip] $out exists"
      continue
    fi
    echo "=== RUN $suite / $pert ==="
    .venv/bin/python -u eval_libero_plus/eval_libero_pro.py \
      --checkpoint "$CHECKPOINT" \
      --suite "$suite" \
      --perturbation "$pert" \
      --n-episodes "$N_EP" \
      --device "$DEVICE" \
      --output "$out"
  done
done

.venv/bin/python eval_libero_plus/aggregate_pro_results.py
