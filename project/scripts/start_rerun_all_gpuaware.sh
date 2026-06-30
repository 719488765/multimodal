#!/usr/bin/env bash
# Author: AI
# Date: 2026-03-31
# Description: 重跑实验总控入口（mainline/expansion/extra/all，支持 auto/single/dual）

set -euo pipefail

PROJECT_DIR="/home/lizhichun_24/sda1/code/multimodal/project"

GROUP="${1:-all}"   # mainline | expansion | extra | all
MODE="${2:-auto}"   # auto | single | dual
DRY_RUN="${3:-0}"   # 0 | 1

MAINLINE_SCRIPT="$PROJECT_DIR/scripts/start_rerun_mainline_gpuaware.sh"
EXPANSION_SCRIPT="$PROJECT_DIR/scripts/start_rerun_expansion_gpuaware.sh"
EXTRA_SCRIPT="$PROJECT_DIR/scripts/start_rerun_extra_dims_gpuaware.sh"

if [[ ! -d "$PROJECT_DIR" ]]; then
  echo "[ERROR] project dir not found: $PROJECT_DIR"
  exit 1
fi

for f in "$MAINLINE_SCRIPT" "$EXPANSION_SCRIPT" "$EXTRA_SCRIPT"; do
  if [[ ! -x "$f" ]]; then
    echo "[ERROR] required executable script missing: $f"
    echo "Please run: chmod +x \"$f\""
    exit 1
  fi
done

if [[ "$MODE" != "auto" && "$MODE" != "single" && "$MODE" != "dual" ]]; then
  echo "[ERROR] unsupported mode: $MODE (use auto|single|dual)"
  exit 1
fi
if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
  echo "[ERROR] unsupported dry-run flag: $DRY_RUN (use 0|1)"
  exit 1
fi

run_group() {
  local name="$1"
  local script="$2"
  echo
  echo "=============================="
  echo "[INFO] Launch group: $name (mode=$MODE dry_run=$DRY_RUN)"
  echo "=============================="
  (cd "$PROJECT_DIR" && "$script" "$MODE" "$DRY_RUN")
}

case "$GROUP" in
  mainline)
    run_group "mainline" "$MAINLINE_SCRIPT"
    ;;
  expansion)
    run_group "expansion" "$EXPANSION_SCRIPT"
    ;;
  extra)
    run_group "extra" "$EXTRA_SCRIPT"
    ;;
  all)
    run_group "mainline" "$MAINLINE_SCRIPT"
    run_group "expansion" "$EXPANSION_SCRIPT"
    run_group "extra" "$EXTRA_SCRIPT"
    ;;
  *)
    echo "[ERROR] unsupported group: $GROUP"
    echo "Usage: ./scripts/start_rerun_all_gpuaware.sh [mainline|expansion|extra|all] [auto|single|dual] [0|1]"
    exit 1
    ;;
esac

echo
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[DONE] dry-run finished. No tmux sessions were started."
else
  echo "[DONE] requested groups launched. Check sessions with: tmux ls"
fi
