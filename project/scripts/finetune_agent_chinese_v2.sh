#!/usr/bin/env bash
# 从当前 agent_chinese best 或 ap2_m1 继续微调 v2（早停、CREMA neutral=4）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/home/lizhichun_24/.conda/envs/myenv310/bin/python}"
CONFIG="config/rerun/accuracy_plan/ap2_M1_chinese_text_agent_v2.yaml"
RESUME="${RESUME:-checkpoints_accuracy_seq/AP2_M1_chinese_text_agent/checkpoint_finetune_best_f1.pth}"

if [[ ! -f "$RESUME" ]]; then
  RESUME="checkpoints_accuracy_seq/AP2_M1_ES_3ds_effbatch8_s3407_20260422_210615/checkpoint_pretrain_best_f1.pth"
fi

echo "==> Finetune agent_chinese v2 from $RESUME"
echo "    Config: $CONFIG"

exec "$PYTHON" scripts/train.py \
  --config "$CONFIG" \
  --mode finetune \
  --resume "$RESUME" \
  --skip_text_encoder
