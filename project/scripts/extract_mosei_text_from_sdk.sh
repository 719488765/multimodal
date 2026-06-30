#!/usr/bin/env bash
# MOSEI 文本从 SDK TimestampedWords.csd 补全
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RAW_DIR="${RAW_DIR:-$ROOT/downloads/CMU_MOSEI_raw/CMU-MOSEI}"
EXTRA=()

if [[ "${1:-}" == "--limit" ]]; then
  EXTRA+=(--limit "$2")
  shift 2
fi

echo "==> MOSEI SDK 文本补全 raw_dir=$RAW_DIR"
python3 scripts/extract_mosei_text_from_sdk.py \
  --data-root "$ROOT/data" \
  --raw-dir "$RAW_DIR" \
  "${EXTRA[@]}" \
  "$@"
