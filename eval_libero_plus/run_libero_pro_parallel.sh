#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

CHECKPOINT="${CHECKPOINT:-checkpoint/stwam_libero_ddp/latest.pt}"
N_EP="${N_EPISODES:-10}"
IFS=',' read -r -a PERTS <<< "${PERTURBATIONS:-object,swap,language,task,environment}"
LOGDIR="eval_libero_plus/logs"
mkdir -p "$LOGDIR"

run_suite() {
  local suite="$1"
  local device="$2"
  for pert in "${PERTS[@]}"; do
    pert="${pert//[[:space:]]/}"
    [[ -n "$pert" ]] || continue
    local out="$LOGDIR/eval_pro_${suite}_${pert}.csv"
    local log="$LOGDIR/eval_pro_${suite}_${pert}.log"
    if [[ -f "$out" ]]; then
      echo "[skip] $out"
      continue
    fi
    : > "$log"
    echo "[$(date +%H:%M:%S)] START $suite/$pert on $device" | tee -a "$log"
    .venv/bin/python -u eval_libero_plus/eval_libero_pro.py \
      --checkpoint "$CHECKPOINT" \
      --suite "$suite" \
      --perturbation "$pert" \
      --n-episodes "$N_EP" \
      --device "$device" \
      --output "$out" 2>&1 | tee -a "$log"
  done
}

run_suite libero_spatial cuda:1 &
run_suite libero_object cuda:2 &
run_suite libero_goal cuda:3 &
run_suite libero_10 cuda:4 &
wait

.venv/bin/python eval_libero_plus/aggregate_pro_results.py
echo "[done] LIBERO-PRO run complete (PERTURBATIONS=${PERTURBATIONS:-object,swap,language,task,environment})"
