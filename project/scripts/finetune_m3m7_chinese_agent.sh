#!/usr/bin/env bash
# M3_M7 V/A/fusion + bert-base-chinese text 微调（中文 Agent 部署）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
source "$ROOT/scripts/r4_env.sh"
PY="${R4_PYTHON:-python3}"

CONFIG="config/sdavt_v3_r4/p3_m3/meld/M3_M7_chinese_agent.yaml"
RESUME="${RESUME:-checkpoints_sdavt_v3_r4/SDAVT_R4_M3_M7_combo/checkpoint_pretrain_best_f1.pth}"

if [[ ! -f "$RESUME" ]]; then
  echo "[FAIL] M3_M7 resume ckpt missing: $RESUME" >&2
  exit 1
fi

echo "==> Finetune M3_M7 Chinese Agent from $RESUME"
echo "    Config: $CONFIG"
echo "    Skipping text_encoder weights (bert-base-chinese init)"

exec "$PY" scripts/train.py \
  --config "$CONFIG" \
  --mode finetune \
  --resume "$RESUME" \
  --skip_text_encoder
