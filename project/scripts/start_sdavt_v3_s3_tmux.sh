#!/usr/bin/env bash
# SDAVT v3 S3 修复重训 — MOSEI + CREMA（MELD S2 已达标，仅 eval）
#
# 用法:
#   ./scripts/start_sdavt_v3_s3_tmux.sh all
#   ./scripts/start_sdavt_v3_s3_tmux.sh mosei|crema|eval_meld|clean_tb

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
ENV_NAME="${ENV_NAME:-myenv310}"
GPU_MOSEI="${GPU_MOSEI:-0}"
GPU_CREMA="${GPU_CREMA:-1}"
PHASE="${1:-all}"

CONDA_SH=""
for _c in \
  "${CONDA_BASE:-}/etc/profile.d/conda.sh" \
  "$HOME/miniconda3/etc/profile.d/conda.sh" \
  "$HOME/anaconda3/etc/profile.d/conda.sh" \
  "/usr/local/miniconda3/etc/profile.d/conda.sh" \
  "/opt/conda/etc/profile.d/conda.sh"; do
  if [[ -f "$_c" ]]; then CONDA_SH="$_c"; break; fi
done
if [[ -z "$CONDA_SH" ]] && command -v conda >/dev/null 2>&1; then
  _b="$(conda info --base 2>/dev/null || true)"
  [[ -n "$_b" && -f "$_b/etc/profile.d/conda.sh" ]] && CONDA_SH="$_b/etc/profile.d/conda.sh"
fi

CFG="config/sdavt_v3"
CMD_MOSEI="python3 scripts/train.py --config ${CFG}/mosei/S3_O0_mosei_AVT_ES_npy_ap1fix.yaml --mode pretrain"
CMD_CREMA="python3 scripts/train.py --config ${CFG}/crema/S3_C0_crema_AV_standard_ap1fix.yaml --mode pretrain"
MELD_CKPT="checkpoints_sdavt_v3/SDAVT_S2_M1_meld_AVT_ES_native_ap1plus_20260614_162353/checkpoint_pretrain_best_f1.pth"
CMD_EVAL_MELD="python3 scripts/eval_checkpoint.py --config ${CFG}/meld/S2_M1_meld_AVT_ES_native_ap1plus.yaml --checkpoint ${MELD_CKPT} --split val"

if [[ "$PHASE" == "clean_tb" ]]; then
  ARCHIVE="$PROJECT_DIR/logs_sdavt_v3_archived/s2_$(date +%Y%m%d)"
  mkdir -p "$ARCHIVE"
  shopt -s nullglob
  for d in "$PROJECT_DIR/logs_sdavt_v3"/SDAVT_S2_*; do
    [[ -d "$d" ]] || continue
    echo "归档: $(basename "$d")"
    mv "$d" "$ARCHIVE/"
  done
  tmux kill-session -t sdavt_tb 2>/dev/null || true
  tmux new-session -d -s sdavt_tb "bash $PROJECT_DIR/scripts/tensorboard_sdavt_v3.sh 6007"
  echo "[OK] S2 已归档至 $ARCHIVE，TensorBoard 重启"
  exit 0
fi

if [[ "$PHASE" == "eval_meld" ]]; then
  cd "$PROJECT_DIR"
  source "$CONDA_SH"
  conda activate "$ENV_NAME"
  $CMD_EVAL_MELD
  exit 0
fi

declare -a RUNS=()
case "$PHASE" in
  all)
    RUNS=(
      "sdavt_s3_o0_mosei:${GPU_MOSEI}:${CMD_MOSEI}"
      "sdavt_s3_c0_crema:${GPU_CREMA}:${CMD_CREMA}"
    )
    ;;
  mosei) RUNS=("sdavt_s3_o0_mosei:${GPU_MOSEI}:${CMD_MOSEI}") ;;
  crema) RUNS=("sdavt_s3_c0_crema:${GPU_CREMA}:${CMD_CREMA}") ;;
  *) echo "Usage: $0 {all|mosei|crema|eval_meld|clean_tb}"; exit 1 ;;
esac

[[ -f "$CONDA_SH" ]] || { echo "[ERROR] conda.sh not found"; exit 1; }

for dead in sdavt_s2_o0_mosei sdavt_s2_c1_crema sdavt_s2_m1_meld; do
  tmux kill-session -t "$dead" 2>/dev/null || true
done

for item in "${RUNS[@]}"; do
  session="${item%%:*}"
  rest="${item#*:}"
  gpu="${rest%%:*}"
  cmd="${rest#*:}"
  tmux kill-session -t "$session" 2>/dev/null || true
  tmux new-session -d -s "$session"
  tmux send-keys -t "$session" "cd \"$PROJECT_DIR\"" C-m
  tmux send-keys -t "$session" "source \"$CONDA_SH\"" C-m
  tmux send-keys -t "$session" "conda activate $ENV_NAME" C-m
  tmux send-keys -t "$session" "export CUDA_VISIBLE_DEVICES=$gpu" C-m
  tmux send-keys -t "$session" "$cmd" C-m
  echo "[OK] $session GPU=$gpu"
done

echo "MELD S2 eval: bash scripts/start_sdavt_v3_s3_tmux.sh eval_meld"
