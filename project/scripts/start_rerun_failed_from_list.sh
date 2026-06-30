#!/usr/bin/env bash
# Author: AI
# Date: 2026-03-31
# Description: 从失败清单补跑任务（支持 dry-run；正式运行会等待完成并生成下一轮失败清单）

set -euo pipefail

PROJECT_DIR="/home/lizhichun_24/sda1/code/multimodal/project"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
ENV_NAME="myenv310"
STATUS_DIR="$PROJECT_DIR/logs_rerun/.launcher_status"
USE_CONDA="${USE_CONDA:-1}"                              # 1 | 0

resolve_conda_sh() {
  if [[ -n "${CONDA_SH:-}" && -f "$CONDA_SH" ]]; then
    return 0
  fi
  if command -v conda >/dev/null 2>&1; then
    local base
    base="$(conda info --base 2>/dev/null || true)"
    if [[ -n "$base" && -f "$base/etc/profile.d/conda.sh" ]]; then
      CONDA_SH="$base/etc/profile.d/conda.sh"
      return 0
    fi
  fi
  local candidates=(
    "$HOME/miniconda3/etc/profile.d/conda.sh"
    "$HOME/anaconda3/etc/profile.d/conda.sh"
    "/opt/miniconda3/etc/profile.d/conda.sh"
    "/opt/anaconda3/etc/profile.d/conda.sh"
  )
  local c
  for c in "${candidates[@]}"; do
    if [[ -f "$c" ]]; then
      CONDA_SH="$c"
      return 0
    fi
  done
  return 1
}

LIST_FILE="${1:-$STATUS_DIR/failed_items_latest.txt}" # 每行格式: session:gpu:cmd
DRY_RUN="${2:-0}"                                      # 0 | 1
POLL_SEC="${3:-20}"                                    # 轮询间隔秒数

if [[ ! -d "$PROJECT_DIR" ]]; then
  echo "[ERROR] project dir not found: $PROJECT_DIR"
  exit 1
fi
if [[ "$USE_CONDA" != "0" && "$DRY_RUN" != "1" ]]; then
  if ! resolve_conda_sh; then
    echo "[ERROR] conda script not found. Tried CONDA_SH and common locations."
    echo "        You can set env manually: export CONDA_SH=/path/to/conda.sh"
    exit 1
  fi
fi
if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
  echo "[ERROR] unsupported dry-run flag: $DRY_RUN (use 0|1)"
  exit 1
fi
if ! [[ "$POLL_SEC" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] POLL_SEC must be positive integer"
  exit 1
fi
if [[ ! -f "$LIST_FILE" ]]; then
  echo "[ERROR] list file not found: $LIST_FILE"
  exit 1
fi

mkdir -p "$STATUS_DIR"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
NEXT_FAILED_ITEMS_FILE="$STATUS_DIR/failed_items_next_${RUN_ID}.txt"
NEXT_FAILED_SESSIONS_FILE="$STATUS_DIR/failed_sessions_next_${RUN_ID}.txt"
LATEST_FAILED_ITEMS_FILE="$STATUS_DIR/failed_items_latest.txt"
LATEST_FAILED_SESSIONS_FILE="$STATUS_DIR/failed_sessions_latest.txt"

declare -a STARTED_SESSIONS=()
declare -a PROCESSED_ITEMS=()

echo "[INFO] list_file=$LIST_FILE dry_run=$DRY_RUN poll_sec=$POLL_SEC"
echo "[INFO] use_conda=$USE_CONDA conda_sh=$CONDA_SH"

while IFS= read -r line || [[ -n "$line" ]]; do
  [[ -z "$line" ]] && continue
  session="${line%%:*}"
  rest="${line#*:}"
  gpu="${rest%%:*}"
  cmd="${rest#*:}"

  PROCESSED_ITEMS+=("$line")

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

  status_file="$STATUS_DIR/${session}.rerun.${RUN_ID}.exitcode"
  rm -f "$status_file"

  tmux new-session -d -s "$session"
  tmux send-keys -t "$session" "cd \"$PROJECT_DIR\"" C-m
  if [[ "$USE_CONDA" != "0" ]]; then
    tmux send-keys -t "$session" "source \"$CONDA_SH\"" C-m
    tmux send-keys -t "$session" "conda activate $ENV_NAME" C-m
  fi
  tmux send-keys -t "$session" "export CUDA_VISIBLE_DEVICES=$gpu" C-m
  tmux send-keys -t "$session" "$cmd; ec=\$?; echo \$ec > \"$status_file\"; exit \$ec" C-m
  STARTED_SESSIONS+=("$session")
  echo "[OK] started $session on CUDA_VISIBLE_DEVICES=$gpu"
done < "$LIST_FILE"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[DONE] dry-run finished. processed items: ${#PROCESSED_ITEMS[@]}"
  exit 0
fi

while true; do
  running=0
  for s in "${STARTED_SESSIONS[@]}"; do
    if tmux has-session -t "$s" 2>/dev/null; then
      running=$((running + 1))
    fi
  done
  if [[ "$running" -eq 0 ]]; then
    break
  fi
  echo "[WAIT] rerun sessions still running=$running, sleep ${POLL_SEC}s..."
  sleep "$POLL_SEC"
done

: > "$NEXT_FAILED_ITEMS_FILE"
: > "$NEXT_FAILED_SESSIONS_FILE"
for item in "${PROCESSED_ITEMS[@]}"; do
  session="${item%%:*}"
  status_file="$STATUS_DIR/${session}.rerun.${RUN_ID}.exitcode"
  if [[ ! -f "$status_file" ]]; then
    echo "[WARN] missing exit code file, treat as failed: $session"
    echo "$item" >> "$NEXT_FAILED_ITEMS_FILE"
    echo "$session" >> "$NEXT_FAILED_SESSIONS_FILE"
    continue
  fi
  code="$(<"$status_file")"
  if [[ "$code" != "0" ]]; then
    echo "[WARN] failed in rerun: $session exit_code=$code"
    echo "$item" >> "$NEXT_FAILED_ITEMS_FILE"
    echo "$session" >> "$NEXT_FAILED_SESSIONS_FILE"
  fi
done

cp "$NEXT_FAILED_ITEMS_FILE" "$LATEST_FAILED_ITEMS_FILE"
cp "$NEXT_FAILED_SESSIONS_FILE" "$LATEST_FAILED_SESSIONS_FILE"

remaining="$(wc -l < "$NEXT_FAILED_ITEMS_FILE" | tr -d ' ')"
echo "[DONE] processed items: ${#PROCESSED_ITEMS[@]}, remaining failed: $remaining"
echo "[INFO] next failed list: $NEXT_FAILED_ITEMS_FILE"
echo "[INFO] latest failed list updated: $LATEST_FAILED_ITEMS_FILE"
