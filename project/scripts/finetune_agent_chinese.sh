#!/usr/bin/env bash
# 从 ap2_m1 预训练权重微调 bert-base-chinese 文本 backbone（三数据集）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/home/lizhichun_24/.conda/envs/myenv310/bin/python}"
CONFIG="config/rerun/accuracy_plan/ap2_M1_chinese_text_agent.yaml"
RESUME="${RESUME:-checkpoints_accuracy_seq/AP2_M1_ES_3ds_effbatch8_s3407_20260422_210615/checkpoint_pretrain_best_f1.pth}"

if [[ ! -f "$RESUME" ]]; then
  echo "ERROR: 未找到 ap2_m1 checkpoint: $RESUME" >&2
  echo "请设置 RESUME= 指向 checkpoint_pretrain_best_f1.pth" >&2
  exit 1
fi

echo "==> Finetune agent_chinese from $RESUME"
echo "    Config: $CONFIG"
echo "    首次运行会下载 bert-base-chinese"

exec "$PYTHON" scripts/train.py \
  --config "$CONFIG" \
  --mode finetune \
  --resume "$RESUME" \
  --skip_text_encoder
