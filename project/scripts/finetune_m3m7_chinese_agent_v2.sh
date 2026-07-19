#!/usr/bin/env bash
# M3_M7 中文 Agent v2：从 v1 finetune best ckpt 二阶段微调
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
source "$ROOT/scripts/r4_env.sh"
PY="${R4_PYTHON:-python3}"

CONFIG="config/sdavt_v3_r4/p3_m3/meld/M3_M7_chinese_agent_v2.yaml"
RESUME="${RESUME:-checkpoints_sdavt_v3_r4/SDAVT_R4_M3_M7_chinese_agent/checkpoint_finetune_best_f1.pth}"
MAX_TRAIN="${MAX_TRAIN_SAMPLES:-2000}"

if [[ ! -f "$RESUME" ]]; then
  echo "[FAIL] v1 finetune ckpt missing: $RESUME" >&2
  exit 1
fi

if [[ "${SKIP_DATA_PREP:-0}" != "1" ]]; then
  bash "$ROOT/scripts/prepare_meld_zh_agent_v2_data.sh"
fi

echo "==> Finetune M3_M7 Chinese Agent v2 from $RESUME"
echo "    Config: $CONFIG"
echo "    max_train_samples: $MAX_TRAIN"

exec "$PY" scripts/train.py \
  --config "$CONFIG" \
  --mode finetune \
  --resume "$RESUME" \
  --init_weights \
  --max_train_samples "$MAX_TRAIN"
