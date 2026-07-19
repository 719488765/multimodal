#!/usr/bin/env bash
# 后台监视 P2 ES 重训，写入 p2_es_retrain_watch.log（可从仓库根目录 tail）
# DISABLED BY DEFAULT — continuous watch loops are obsolete after R4 close-out.
# Opt-in: ENABLE_R4_WATCH=1 bash scripts/watch_r4_p2_es_retrain.sh
set -euo pipefail

if [[ "${ENABLE_R4_WATCH:-0}" != "1" ]]; then
  echo "[watch_r4_p2_es] DISABLED (default). Set ENABLE_R4_WATCH=1 to run."
  exit 0
fi

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"

WATCH_LOG="outputs_sdavt_v3_r4/status/p2_es_retrain_watch.log"
mkdir -p "$(dirname "$WATCH_LOG")"

while true; do
  {
    echo "===== $(date '+%Y-%m-%d %H:%M:%S') ====="
    bash scripts/tail_r4_training.sh status 2>&1 | sed 's/^/  /'
  } >> "$WATCH_LOG"
  sleep 300
done
