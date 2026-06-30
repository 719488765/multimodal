#!/usr/bin/env bash
# MELD 单域 Agent 配方训练（推荐替代三混合部署）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/home/lizhichun_24/.conda/envs/myenv310/bin/python}"
CONFIG="config/rerun/accuracy_plan/ap2_M1_meld_only_agent.yaml"
RESUME="${RESUME:-}"
SKIP_AUDIO_EXTRACT="${SKIP_AUDIO_EXTRACT:-0}"

# 训练前检查 / 提取 MELD 音频
if [[ "$SKIP_AUDIO_EXTRACT" != "1" ]]; then
  echo "==> 检查 MELD 音频 (data/*/audio/meld_*.wav)"
  PENDING="$("$PYTHON" scripts/extract_meld_audio.py --dry-run 2>/dev/null | sed -n 's/.*待提取: \([0-9]*\).*/\1/p' | head -1)"
  PENDING="${PENDING:-999999}"
  if [[ "$PENDING" != "0" ]]; then
    echo "==> 开始从 mp4 提取音频（约 15–40 分钟，8 并行，backend=PyAV）"
    chmod +x scripts/extract_meld_audio.sh
    ./scripts/extract_meld_audio.sh
  else
    echo "    MELD wav 已齐全，跳过提取"
  fi
fi

ARGS=(--config "$CONFIG" --mode pretrain)
if [[ -n "$RESUME" ]]; then
  ARGS+=(--resume "$RESUME")
fi

echo "==> Train MELD-only agent model"
echo "    Config: $CONFIG"
echo "    完成后: ./scripts/apply_deploy_preset.sh meld_agent"
echo "    （需先将 best ckpt 链到 preset 目录，见训练日志目录名）"

exec "$PYTHON" scripts/train.py "${ARGS[@]}"
