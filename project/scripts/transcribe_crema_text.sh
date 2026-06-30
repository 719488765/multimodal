#!/usr/bin/env bash
# CREMA 文本 ASR 补全（Whisper）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DEVICE="${DEVICE:-cuda}"
ENGINE="${ENGINE:-transformers}"
MODEL="${MODEL:-base}"
LIMIT="${LIMIT:-0}"
EXTRA=()

if [[ "${1:-}" == "--limit" ]]; then
  LIMIT="$2"
  shift 2
fi
if [[ "$LIMIT" != "0" ]]; then
  EXTRA+=(--limit "$LIMIT")
fi

echo "==> CREMA ASR 文本补全 (device=$DEVICE model=$MODEL)"
python3 scripts/transcribe_crema_text.py \
  --data-root "$ROOT/data" \
  --engine "$ENGINE" \
  --model "$MODEL" \
  --device "$DEVICE" \
  "${EXTRA[@]}" \
  "$@"
