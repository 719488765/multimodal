#!/usr/bin/env bash
# SDAVT v3 S2 优化重训 — MELD / CREMA / MOSEI 分域
#
# 用法:
#   ./scripts/start_sdavt_v3_s2_tmux.sh all      # GPU0 MELD + GPU1 CREMA + GPU0 MOSEI(小batch)
#   ./scripts/start_sdavt_v3_s2_tmux.sh retry    # 同 all（MELD+CREMA，不含 MOSEI）
#   ./scripts/start_sdavt_v3_s2_tmux.sh mosei    # 仅 MOSEI S2-O0
#   ./scripts/start_sdavt_v3_s2_tmux.sh meld
#   ./scripts/start_sdavt_v3_s2_tmux.sh crema
#   ./scripts/start_sdavt_v3_s2_tmux.sh clean_tb # 归档 S1 日志并重启 TensorBoard

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
ENV_NAME="${ENV_NAME:-myenv310}"
GPU_MELD="${GPU_MELD:-0}"
GPU_CREMA="${GPU_CREMA:-1}"
GPU_MOSEI="${GPU_MOSEI:-0}"
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
CMD_MELD="python3 scripts/train.py --config ${CFG}/meld/S2_M1_meld_AVT_ES_native_ap1plus.yaml --mode pretrain"
CMD_CREMA="python3 scripts/train.py --config ${CFG}/crema/S2_C1_crema_AV_ES_no_text.yaml --mode pretrain"
CMD_MOSEI="python3 scripts/train.py --config ${CFG}/mosei/S2_O0_mosei_AVT_ES_npy_ap1plus.yaml --mode pretrain"

if [[ "$PHASE" == "clean_tb" ]]; then
  bash "$PROJECT_DIR/scripts/clean_sdavt_v3_logs_s2.sh"
  tmux kill-session -t sdavt_tb 2>/dev/null || true
  tmux new-session -d -s sdavt_tb "bash $PROJECT_DIR/scripts/tensorboard_sdavt_v3.sh 6007"
  echo "[OK] TensorBoard 已重启，仅展示 logs_sdavt_v3/ 下 S2 run"
  exit 0
fi

declare -a RUNS=()
case "$PHASE" in
  all)
    RUNS=(
      "sdavt_s2_m1_meld:${GPU_MELD}:${CMD_MELD}"
      "sdavt_s2_c1_crema:${GPU_CREMA}:${CMD_CREMA}"
      "sdavt_s2_o0_mosei:${GPU_MOSEI}:${CMD_MOSEI}"
    )
    ;;
  retry)
    RUNS=(
      "sdavt_s2_m1_meld:${GPU_MELD}:${CMD_MELD}"
      "sdavt_s2_c1_crema:${GPU_CREMA}:${CMD_CREMA}"
    )
    ;;
  mosei) RUNS=("sdavt_s2_o0_mosei:${GPU_MOSEI}:${CMD_MOSEI}") ;;
  meld)  RUNS=("sdavt_s2_m1_meld:${GPU_MELD}:${CMD_MELD}") ;;
  crema) RUNS=("sdavt_s2_c1_crema:${GPU_CREMA}:${CMD_CREMA}") ;;
  list)
    echo "$CMD_MELD"
    echo "$CMD_CREMA"
    echo "$CMD_MOSEI"
    exit 0
    ;;
  *) echo "Usage: $0 {all|retry|mosei|meld|crema|clean_tb|list}"; exit 1 ;;
esac

[[ -f "$CONDA_SH" ]] || { echo "[ERROR] conda.sh not found"; exit 1; }

# 停止错误的 S1 MOSEI 会话
for dead in sdavt_o0_mosei sdavt_m1_meld sdavt_c0_crema; do
  if tmux has-session -t "$dead" 2>/dev/null; then
    echo "[INFO] 停止旧会话: $dead"
    tmux send-keys -t "$dead" C-c 2>/dev/null || true
    sleep 1
    tmux kill-session -t "$dead" 2>/dev/null || true
  fi
done

for item in "${RUNS[@]}"; do
  session="${item%%:*}"
  rest="${item#*:}"
  gpu="${rest%%:*}"
  cmd="${rest#*:}"

  if tmux has-session -t "$session" 2>/dev/null; then
    echo "[INFO] session exists, skip: $session"
    continue
  fi

  tmux new-session -d -s "$session"
  tmux send-keys -t "$session" "cd \"$PROJECT_DIR\"" C-m
  tmux send-keys -t "$session" "source \"$CONDA_SH\"" C-m
  tmux send-keys -t "$session" "conda activate $ENV_NAME" C-m
  tmux send-keys -t "$session" "export CUDA_VISIBLE_DEVICES=$gpu" C-m
  tmux send-keys -t "$session" "$cmd" C-m
  echo "[OK] $session GPU=$gpu"
done

echo
echo "TensorBoard 清理 S1: bash scripts/start_sdavt_v3_s2_tmux.sh clean_tb"
