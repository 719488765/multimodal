#!/usr/bin/env bash
# Author: AI
# Date: 2026-03-31
# Description: 按单卡/双卡策略启动主线重跑（支持 CUDA_VISIBLE_DEVICES 绑定）

set -euo pipefail

PROJECT_DIR="/home/lizhichun_24/sda1/code/multimodal/project"
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
ENV_NAME="myenv310"

MODE="${1:-auto}"  # auto | single | dual
DRY_RUN="${2:-0}"  # 0 | 1
GPU_SINGLE="${GPU_SINGLE:-0}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
MIN_FREE_MB="${MIN_FREE_MB:-12000}"

if [[ ! -d "$PROJECT_DIR" ]]; then
  echo "[ERROR] project dir not found: $PROJECT_DIR"
  exit 1
fi

if [[ "$DRY_RUN" != "1" && ! -f "$CONDA_SH" ]]; then
  echo "[ERROR] conda script not found: $CONDA_SH"
  exit 1
fi

free0=0
free1=0
if command -v nvidia-smi >/dev/null 2>&1; then
  free0=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n '1p' | tr -d ' ')
  free1=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n '2p' | tr -d ' ')
fi

if [[ "$MODE" == "auto" ]]; then
  if [[ "${free0:-0}" -ge "$MIN_FREE_MB" && "${free1:-0}" -ge "$MIN_FREE_MB" ]]; then
    MODE="dual"
  else
    MODE="single"
  fi
fi

echo "[INFO] mode=$MODE free_mem_mb(gpu0,gpu1)=(${free0:-NA},${free1:-NA})"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[INFO] dry-run enabled: only print plan, no tmux start"
fi

declare -a RUNS_SINGLE=(
  "rerun_at_noda:${GPU_SINGLE}:python3 scripts/train.py --config config/rerun/config_AT_noDA.yaml --mode pretrain"
  "rerun_at_da:${GPU_SINGLE}:python3 scripts/train.py --config config/rerun/config_AT_DA.yaml --mode pretrain"
  "rerun_vt_noda:${GPU_SINGLE}:python3 scripts/train.py --config config/rerun/config_VT_noDA.yaml --mode pretrain"
  "rerun_avt_noda:${GPU_SINGLE}:python3 scripts/train.py --config config/rerun/config_AVT_noDA.yaml --mode pretrain"
  "rerun_avt_da:${GPU_SINGLE}:python3 scripts/train.py --config config/rerun/config_AVT_DA.yaml --mode pretrain"
  "rerun_avt_es:${GPU_SINGLE}:python3 scripts/train.py --config config/rerun/config_AVT_noDA_emotion_shift.yaml --mode pretrain"
)

declare -a RUNS_DUAL=(
  "rerun_at_noda:${GPU0}:python3 scripts/train.py --config config/rerun/config_AT_noDA.yaml --mode pretrain"
  "rerun_at_da:${GPU1}:python3 scripts/train.py --config config/rerun/config_AT_DA.yaml --mode pretrain"
  "rerun_vt_noda:${GPU0}:python3 scripts/train.py --config config/rerun/config_VT_noDA.yaml --mode pretrain"
  "rerun_avt_noda:${GPU1}:python3 scripts/train.py --config config/rerun/config_AVT_noDA.yaml --mode pretrain"
  "rerun_avt_da:${GPU0}:python3 scripts/train.py --config config/rerun/config_AVT_DA.yaml --mode pretrain"
  "rerun_avt_es:${GPU1}:python3 scripts/train.py --config config/rerun/config_AVT_noDA_emotion_shift.yaml --mode pretrain"
)

if [[ "$MODE" == "single" ]]; then
  RUNS=("${RUNS_SINGLE[@]}")
elif [[ "$MODE" == "dual" ]]; then
  RUNS=("${RUNS_DUAL[@]}")
else
  echo "[ERROR] unsupported mode: $MODE (use auto|single|dual)"
  exit 1
fi

for item in "${RUNS[@]}"; do
  session="${item%%:*}"
  rest="${item#*:}"
  gpu="${rest%%:*}"
  cmd="${rest#*:}"

  if [[ "$DRY_RUN" == "1" ]]; then
    if tmux has-session -t "$session" 2>/dev/null; then
      echo "[PLAN] skip existing session: $session"
    else
      echo "[PLAN] start $session on CUDA_VISIBLE_DEVICES=$gpu"
      echo "       cmd: $cmd"
    fi
    continue
  fi

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
  echo "[OK] started $session on CUDA_VISIBLE_DEVICES=$gpu"
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry-run done."
else
  echo "Done. Check with: tmux ls"
fi
