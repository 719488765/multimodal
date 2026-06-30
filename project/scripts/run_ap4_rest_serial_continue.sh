#!/usr/bin/env bash
# Resume AP4 w005 from checkpoint, then run remaining AP4 configs serially in one tmux session.
# Usage: GPU_ID=0 ./scripts/run_ap4_rest_serial_continue.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

ENV_NAME="${ENV_NAME:-myenv310}"
GPU_ID="${GPU_ID:-0}"
CFG_DIR="config/rerun/accuracy_plan"

CONDA_SH="${CONDA_SH:-}"
if [[ -z "$CONDA_SH" && -n "${CONDA_BASE:-}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"
fi
if [[ -z "$CONDA_SH" ]] && command -v conda >/dev/null 2>&1; then
  _b="$(conda info --base 2>/dev/null || true)"
  [[ -n "$_b" && -f "$_b/etc/profile.d/conda.sh" ]] && CONDA_SH="$_b/etc/profile.d/conda.sh"
fi
if [[ -z "$CONDA_SH" || ! -f "$CONDA_SH" ]]; then
  echo "[ERROR] conda.sh not found. Set CONDA_SH or CONDA_BASE."
  exit 1
fi

# shellcheck source=/dev/null
source "$CONDA_SH"
conda activate "$ENV_NAME"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
# Prefer local HuggingFace cache (hf-mirror DNS may be unavailable on cluster nodes).
unset HF_ENDPOINT
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

log() { echo "[$(date '+%F %T')] $*"; }

run_train() {
  local cfg="$1"
  local resume="${2:-}"
  log "START config=$cfg resume=${resume:-none} GPU=$GPU_ID"
  if [[ -n "$resume" ]]; then
    python3 scripts/train.py --config "$cfg" --mode pretrain --resume "$resume"
  else
    python3 scripts/train.py --config "$cfg" --mode pretrain
  fi
  log "DONE config=$cfg"
}

W005_RUN="AP4_AVT_pretrain_3datasets_DA_w005_20260514_071550"
W005_CKPT="checkpoints_accuracy_seq/${W005_RUN}/checkpoint_pretrain_epoch_24.pth"

if [[ ! -f "$W005_CKPT" ]]; then
  echo "[ERROR] Missing checkpoint: $W005_CKPT"
  exit 1
fi

log "=== AP4 serial continue: resume w005 then w010, w05lr, uni, s34 ==="

export MULTIMODAL_LOG_RUN_DIR_NAME="$W005_RUN"
run_train "${CFG_DIR}/ap4_config_AVT_DA_w005_accuracy_seq.yaml" "$W005_CKPT"
unset MULTIMODAL_LOG_RUN_DIR_NAME

for cfg in \
  ap4_config_AVT_DA_w010_accuracy_seq.yaml \
  ap4_config_AVT_DA_w005_lr5e5_accuracy_seq.yaml \
  ap4_config_AVT_DA_uniform_accuracy_seq.yaml \
  ap4_config_AVT_DA_seed3407_accuracy_seq.yaml
do
  run_train "${CFG_DIR}/${cfg}"
done

log "=== AP4 rest serial pipeline finished ==="
