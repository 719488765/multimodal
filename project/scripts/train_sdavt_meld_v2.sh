#!/usr/bin/env bash
# SDAVT v3 MELD S1-M1 v2 配方训练（早停 + leader_text + uniform 采样）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
CONFIG="${CONFIG:-config/sdavt_v3/meld/S1_M1_AVT_ES_v2recipe.yaml}"
RESUME="${RESUME:-}"
SKIP_AUDIO_EXTRACT="${SKIP_AUDIO_EXTRACT:-1}"

if [[ "$SKIP_AUDIO_EXTRACT" != "1" ]]; then
  echo "==> 检查 MELD 音频"
  PENDING="$("$PYTHON" scripts/extract_meld_audio.py --dry-run 2>/dev/null | sed -n 's/.*待提取: \([0-9]*\).*/\1/p' | head -1)"
  PENDING="${PENDING:-0}"
  if [[ "$PENDING" != "0" ]]; then
    ./scripts/extract_meld_audio.sh
  fi
fi

ARGS=(--config "$CONFIG" --mode pretrain)
if [[ -n "$RESUME" ]]; then
  ARGS+=(--resume "$RESUME")
fi

echo "==> SDAVT v3 Train MELD S1-M1 (v2 recipe)"
echo "    Config: $CONFIG"
echo "    Logs: logs_sdavt_v3/"
echo "    评估: python3 scripts/eval_checkpoint.py --config $CONFIG --checkpoint <best_f1.pth>"

exec "$PYTHON" scripts/train.py "${ARGS[@]}"
