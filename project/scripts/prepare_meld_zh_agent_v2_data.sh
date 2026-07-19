#!/usr/bin/env bash
# 准备 M3_M7 中文 Agent v2 训练数据：zh 文本增强 + agent_capture 注入
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
source "$ROOT/scripts/r4_env.sh"
PY="${R4_PYTHON:-python3}"

echo "==> [1/3] MELD 中文文本增强"
"$PY" scripts/inject_meld_zh_text_augment.py --limit 500

echo "==> [2/3] 从 benchmark 种子 agent_capture（目标 100 条）"
"$PY" scripts/seed_agent_capture_from_benchmark.py --target-count 100

echo "==> [3/3] 注入 agent_capture 到 data/train"
"$PY" scripts/inject_agent_capture_into_meld.py

TRAIN_N=$(find data/agent_capture/train/labels -name '*.txt' 2>/dev/null | wc -l)
INJECT_N=$(find data/train/labels -name 'meld_train_acap_*' 2>/dev/null | wc -l)
echo "agent_capture train: $TRAIN_N | injected into meld train: $INJECT_N"
echo "Data pipeline ready for M3_M7_chinese_agent_v2"
