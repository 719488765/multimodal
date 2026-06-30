#!/usr/bin/env bash
# MELD 单域 Agent v2 训练（早停 + leader_text + uniform 采样）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/home/lizhichun_24/.conda/envs/myenv310/bin/python}"
CONFIG="${CONFIG:-config/rerun/accuracy_plan/ap2_M1_meld_only_agent_v2.yaml}"
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

echo "==> Train MELD-only agent v2"
echo "    Config: $CONFIG"
echo "    早停/leader_text/uniform 见 yaml"
echo "    评估: python3 scripts/eval_meld_checkpoint.py --config $CONFIG --checkpoint <best_f1.pth>"

exec "$PYTHON" scripts/train.py "${ARGS[@]}"
