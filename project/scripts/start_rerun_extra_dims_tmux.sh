#!/usr/bin/env bash
# Author: AI
# Date: 2026-03-31
# Description: 一键启动新增维度对照实验（uniform/lr/seed）

set -euo pipefail

PROJECT_DIR="/home/lizhichun_24/sda1/code/multimodal/project"
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
ENV_NAME="myenv310"

declare -a RUNS=(
  "rerun_avt_noda_uniform:python3 scripts/train.py --config config/rerun/config_AVT_noDA_uniform.yaml --mode pretrain"
  "rerun_avt_da_uniform:python3 scripts/train.py --config config/rerun/config_AVT_DA_uniform.yaml --mode pretrain"
  "rerun_avt_noda_lr5e5:python3 scripts/train.py --config config/rerun/config_AVT_noDA_lr5e5.yaml --mode pretrain"
  "rerun_avt_da_w005_lr5e5:python3 scripts/train.py --config config/rerun/config_AVT_DA_w005_lr5e5.yaml --mode pretrain"
  "rerun_avt_noda_seed3407:python3 scripts/train.py --config config/rerun/config_AVT_noDA_seed3407.yaml --mode pretrain"
  "rerun_avt_da_seed3407:python3 scripts/train.py --config config/rerun/config_AVT_DA_seed3407.yaml --mode pretrain"
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
echo "Extra-dimension sessions prepared."
echo "Check status with: tmux ls"
