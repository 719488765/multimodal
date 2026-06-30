#!/usr/bin/env bash
# Author: AI
# Date: 2026-03-31
# Description: 循环补跑失败清单直到清零或达到最大轮次

set -euo pipefail

PROJECT_DIR="/home/lizhichun_24/sda1/code/multimodal/project"
STATUS_DIR="$PROJECT_DIR/logs_rerun/.launcher_status"
RUN_SCRIPT="$PROJECT_DIR/scripts/start_rerun_failed_from_list.sh"

LIST_FILE="${1:-$STATUS_DIR/failed_items_latest.txt}"
MAX_ROUNDS="${2:-5}"   # 最多循环轮次
POLL_SEC="${3:-20}"    # 单轮内部轮询间隔（透传）
DRY_RUN="${4:-0}"      # 0 | 1

if [[ ! -x "$RUN_SCRIPT" ]]; then
  echo "[ERROR] run script not executable: $RUN_SCRIPT"
  exit 1
fi
if ! [[ "$MAX_ROUNDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] MAX_ROUNDS must be positive integer"
  exit 1
fi
if ! [[ "$POLL_SEC" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] POLL_SEC must be positive integer"
  exit 1
fi
if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
  echo "[ERROR] DRY_RUN must be 0 or 1"
  exit 1
fi

echo "[INFO] list_file=$LIST_FILE max_rounds=$MAX_ROUNDS poll_sec=$POLL_SEC dry_run=$DRY_RUN"
echo "[INFO] use_conda=${USE_CONDA:-1} conda_sh=${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"

round=1
while [[ "$round" -le "$MAX_ROUNDS" ]]; do
  if [[ ! -f "$LIST_FILE" ]]; then
    echo "[INFO] list file not found, treat as cleared: $LIST_FILE"
    exit 0
  fi

  remaining_before="$(wc -l < "$LIST_FILE" | tr -d ' ')"
  if [[ "$remaining_before" == "0" ]]; then
    echo "[DONE] failure list already empty before round $round"
    exit 0
  fi

  echo
  echo "=============================="
  echo "[INFO] rerun round=$round remaining_before=$remaining_before"
  echo "=============================="

  bash "$RUN_SCRIPT" "$LIST_FILE" "$DRY_RUN" "$POLL_SEC"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DONE] dry-run mode, stop after first round."
    exit 0
  fi

  LIST_FILE="$STATUS_DIR/failed_items_latest.txt"
  if [[ ! -f "$LIST_FILE" ]]; then
    echo "[WARN] latest failed list missing after round $round, stop."
    exit 1
  fi

  remaining_after="$(wc -l < "$LIST_FILE" | tr -d ' ')"
  echo "[INFO] round=$round remaining_after=$remaining_after"
  if [[ "$remaining_after" == "0" ]]; then
    echo "[DONE] failures cleared after round $round"
    exit 0
  fi

  round=$((round + 1))
done

echo "[WARN] reached max rounds=$MAX_ROUNDS, failures still exist:"
echo "       $STATUS_DIR/failed_items_latest.txt"
exit 0
