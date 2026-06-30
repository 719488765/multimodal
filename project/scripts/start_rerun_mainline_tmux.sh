#!/usr/bin/env bash
# Author: AI
# Date: 2026-04-03
# Description: 一键创建并启动主线6组重跑实验 tmux 会话

set -euo pipefail

PROJECT_DIR="/home/lizhichun_24/sda1/code/multimodal/project"
ENV_NAME="myenv310"

# conda.sh：支持自定义根目录（export CONDA_BASE=/path/to/conda），否则按常见路径与 conda info 探测
CONDA_SH=""
if [[ -n "${CONDA_BASE:-}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"
else
  for _c in \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "$HOME/miniforge3/etc/profile.d/conda.sh" \
    "/usr/local/miniconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"; do
    if [[ -f "$_c" ]]; then
      CONDA_SH="$_c"
      break
    fi
  done
fi
if [[ -z "$CONDA_SH" ]] && command -v conda >/dev/null 2>&1; then
  _base="$(conda info --base 2>/dev/null || true)"
  if [[ -n "$_base" && -f "$_base/etc/profile.d/conda.sh" ]]; then
    CONDA_SH="$_base/etc/profile.d/conda.sh"
  fi
fi

if [[ ! -d "$PROJECT_DIR" ]]; then
  echo "[ERROR] PROJECT_DIR not found: $PROJECT_DIR"
  exit 1
fi

if [[ -z "$CONDA_SH" || ! -f "$CONDA_SH" ]]; then
  echo "[ERROR] conda init script not found. Install conda or set CONDA_BASE to the conda root (parent of etc/profile.d)."
  exit 1
fi

declare -a RUNS=(
  "rerun_at_noda:python3 scripts/train.py --config config/rerun/config_AT_noDA.yaml --mode pretrain"
  "rerun_at_da:python3 scripts/train.py --config config/rerun/config_AT_DA.yaml --mode pretrain"
  "rerun_vt_noda:python3 scripts/train.py --config config/rerun/config_VT_noDA.yaml --mode pretrain"
  "rerun_avt_noda:python3 scripts/train.py --config config/rerun/config_AVT_noDA.yaml --mode pretrain"
  "rerun_avt_da:python3 scripts/train.py --config config/rerun/config_AVT_DA.yaml --mode pretrain"
  "rerun_avt_es:python3 scripts/train.py --config config/rerun/config_AVT_noDA_emotion_shift.yaml --mode pretrain"
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
echo "All requested sessions are prepared."
echo "Check status with: tmux ls"
echo "Attach with: tmux attach -t rerun_at_noda"
