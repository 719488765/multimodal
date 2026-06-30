#!/usr/bin/env bash
# MELD v2 A/B：依次训练 v2 基线 与 v2_focal（早停）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/home/lizhichun_24/.conda/envs/myenv310/bin/python}"
export SKIP_AUDIO_EXTRACT=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"

LOG_DIR="$ROOT/logs_accuracy_seq"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/meld_ab_experiments_summary.txt"

run_one() {
  local cfg="$1"
  local tag="$2"
  local log="$LOG_DIR/meld_ab_${tag}.log"
  echo "==> [$tag] config=$cfg"
  CONFIG="$cfg" "$ROOT/scripts/train_meld_agent_v2.sh" 2>&1 | tee "$log"
  local run_dir
  run_dir="$(ls -td "$ROOT/checkpoints_accuracy_seq"/AP2_M1_meld_only_agent_${tag}_* 2>/dev/null | head -1 || true)"
  local ckpt="${run_dir}/checkpoint_pretrain_best_f1.pth"
  if [[ -f "$ckpt" ]]; then
    echo "--- eval $tag ---" | tee -a "$SUMMARY"
    "$PYTHON" scripts/eval_meld_checkpoint.py \
      --config "$cfg" \
      --checkpoint "$ckpt" \
      --split val \
      --output "$LOG_DIR/meld_eval_${tag}.json" 2>&1 | tee -a "$SUMMARY"
  else
    echo "WARN: no best_f1 ckpt for $tag" | tee -a "$SUMMARY"
  fi
}

: > "$SUMMARY"
run_one "config/rerun/accuracy_plan/ap2_M1_meld_only_agent_v2.yaml" "v2"
run_one "config/rerun/accuracy_plan/ap2_M1_meld_only_agent_v2_focal.yaml" "v2_focal"

echo "==> A/B 完成，汇总: $SUMMARY"
