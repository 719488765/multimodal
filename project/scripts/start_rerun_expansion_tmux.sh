#!/usr/bin/env bash
# Author: AI
# Date: 2026-03-31
# Description: 一键创建并启动扩展实验 tmux 会话（T/A/V-only + AVT_DA权重网格）

set -euo pipefail

PROJECT_DIR="/home/lizhichun_24/sda1/code/multimodal/project"
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
ENV_NAME="myenv310"

if [[ ! -d "$PROJECT_DIR" ]]; then
  echo "[ERROR] PROJECT_DIR not found: $PROJECT_DIR"
  exit 1
fi

if [[ ! -f "$CONDA_SH" ]]; then
  echo "[ERROR] conda init script not found: $CONDA_SH"
  exit 1
fi

declare -a RUNS=(
  "rerun_t_only:python3 scripts/train.py --config config/rerun/config_text_only.yaml --mode pretrain"
  "rerun_a_only:python3 scripts/train.py --config config/rerun/config_audio_only.yaml --mode pretrain"
  "rerun_v_only:python3 scripts/train.py --config config/rerun/config_video_only.yaml --mode pretrain"
  "rerun_avt_da_w002:python3 scripts/train.py --config config/rerun/config_AVT_DA_w002.yaml --mode pretrain"
  "rerun_avt_da_w005:python3 scripts/train.py --config config/rerun/config_AVT_DA_w005.yaml --mode pretrain"
  "rerun_avt_da_w010:python3 scripts/train.py --config config/rerun/config_AVT_DA_w010.yaml --mode pretrain"
)

for item in "${RUNS[@]}"; do
  session="${item%%:*}"
  cmd="${item#*:}"

  if tmux has-session -t "$session" 2>/dev/null; then
    echo "[INFO] session exists, skip: $session"
    continue
  fi

  tmux new-session -d -s "$session"
  tmux send-keys -t "$session" "cd \"$PROJECT_DIR\"" C-m
  tmux send-keys -t "$session" "source \"$CONDA_SH\"" C-m
  tmux send-keys -t "$session" "conda activate $ENV_NAME" C-m
  tmux send-keys -t "$session" "$cmd" C-m
  echo "[OK] started session: $session"
done

echo
echo "All requested expansion sessions are prepared."
echo "Check status with: tmux ls"
echo "Attach with: tmux attach -t rerun_t_only"
