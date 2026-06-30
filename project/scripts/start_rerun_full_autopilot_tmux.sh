#!/usr/bin/env bash
# Author: AI
# Date: 2026-03-31
# Description: 在 tmux 中守护启动全自动重跑（断线不断训，重复执行不重复拉起 controller）

set -euo pipefail

PROJECT_DIR="/home/lizhichun_24/sda1/code/multimodal/project"
AUTOPILOT_SCRIPT="$PROJECT_DIR/scripts/start_rerun_full_autopilot.sh"
SESSION_NAME="${SESSION_NAME:-rerun_autopilot}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-myenv310}"
USE_CONDA="${USE_CONDA:-1}"

if [[ ! -x "$AUTOPILOT_SCRIPT" ]]; then
  echo "[ERROR] missing executable script: $AUTOPILOT_SCRIPT"
  exit 1
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "[INFO] controller session already running: $SESSION_NAME"
  echo "[INFO] attach with: tmux attach -t $SESSION_NAME"
  exit 0
fi

tmux new-session -d -s "$SESSION_NAME"
tmux send-keys -t "$SESSION_NAME" "cd \"$PROJECT_DIR\"" C-m
if [[ "$USE_CONDA" != "0" ]]; then
  # 兼容不同 conda 安装路径
  tmux send-keys -t "$SESSION_NAME" "if [ -f \"$CONDA_SH\" ]; then source \"$CONDA_SH\"; elif command -v conda >/dev/null 2>&1; then source \"\$(conda info --base)/etc/profile.d/conda.sh\"; fi" C-m
  tmux send-keys -t "$SESSION_NAME" "conda activate \"$ENV_NAME\"" C-m
fi
tmux send-keys -t "$SESSION_NAME" "MIN_RUN_FREE_MB=\"${MIN_RUN_FREE_MB:-5000}\" AUTO_RERUN_FAILED=\"${AUTO_RERUN_FAILED:-0}\" FORCE_RERUN=\"${FORCE_RERUN:-0}\" \"$AUTOPILOT_SCRIPT\" \"$@\"" C-m

echo "[OK] controller started in tmux session: $SESSION_NAME"
echo "[INFO] attach with: tmux attach -t $SESSION_NAME"
echo "[INFO] check sessions: tmux ls"
