#!/usr/bin/env bash
# Agent 中文采集闭环：整理数据 → 可选 M3_M7 中文微调
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SRC="${1:-}"
if [[ -z "$SRC" ]]; then
  echo "用法: $0 /path/to/raw_recordings [--finetune]" >&2
  echo "参见 data/agent_capture/README.md" >&2
  exit 1
fi

FINETUNE=0
if [[ "${2:-}" == "--finetune" ]]; then
  FINETUNE=1
fi

"$ROOT/scripts/organize_agent_capture.py" --src "$SRC" --split train

TRAIN_N=$(find data/agent_capture/train/labels -name '*.txt' 2>/dev/null | wc -l)
echo "agent_capture train samples: $TRAIN_N"
if [[ "$TRAIN_N" -lt 50 ]]; then
  echo "WARN: 建议每类 ≥15 条、总计 ≥50 条后再 finetune（目标 ≥100）"
fi

if [[ "$FINETUNE" == "1" ]]; then
  bash "$ROOT/scripts/finetune_m3m7_chinese_agent.sh"
fi
