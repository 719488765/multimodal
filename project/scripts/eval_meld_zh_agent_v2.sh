#!/usr/bin/env bash
# 验收 sdavt_meld_zh_agent_v2：离线 benchmark + checkpoint val F1
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
source "$ROOT/scripts/r4_env.sh"
PY="${R4_PYTHON:-python3}"

PRESET="${PRESET:-sdavt_meld_zh_agent_v2}"
V1_CKPT="checkpoints_sdavt_v3_r4/SDAVT_R4_M3_M7_chinese_agent/checkpoint_finetune_best_f1.pth"
V2_CKPT="checkpoints_sdavt_v3_r4/SDAVT_R4_M3_M7_chinese_agent_v2/checkpoint_finetune_best_f1.pth"
V2_CONFIG="config/sdavt_v3_r4/p3_m3/meld/M3_M7_chinese_agent_v2.yaml"

echo "==> [1/4] 离线校准 benchmark（无需 GPU）"
"$PY" scripts/eval_zh_agent_benchmark.py
"$PY" scripts/eval_agent_capture_cases.py

echo ""
echo "==> [2/4] v1 checkpoint val（对照）"
if [[ -f "$V1_CKPT" ]]; then
  "$PY" scripts/eval_meld_checkpoint.py \
    --config config/sdavt_v3_r4/p3_m3/meld/M3_M7_chinese_agent.yaml \
    --checkpoint "$V1_CKPT" \
    --split val \
    --batch-size 1 \
    --out-dir outputs_sdavt_v3_r4/eval_v1_ref || true
fi

echo ""
echo "==> [3/4] v2 checkpoint val"
if [[ ! -f "$V2_CKPT" ]]; then
  echo "[SKIP] v2 ckpt 不存在: $V2_CKPT"
  echo "       请先运行: bash scripts/finetune_m3m7_chinese_agent_v2.sh"
  exit 2
fi

"$PY" scripts/eval_meld_checkpoint.py \
  --config "$V2_CONFIG" \
  --checkpoint "$V2_CKPT" \
  --split val \
  --batch-size 1 \
  --out-dir outputs_sdavt_v3_r4/eval_v2

echo ""
echo "==> [4/4] E2E infer-upload（需 backend :8000）"
if curl -sf http://127.0.0.1:8000/api/v1/health >/dev/null 2>&1; then
  for text in "我很难过" "I am so sad today"; do
    echo "--- text=$text preset=$PRESET ---"
    curl -sf -X POST http://127.0.0.1:8000/api/v1/infer-upload \
      -F "session_id=eval_v2_$(date +%s)" \
      -F "text=$text" \
      -F "metadata={\"checkpoint_preset\":\"$PRESET\",\"degraded_mode\":true}" \
      | "$PY" -c "import sys,json; d=json.load(sys.stdin); e=d.get('emotion',d); print('label=',e.get('final_emotion_label') or e.get('emotion_label'), 'preset=',e.get('checkpoint_preset'), 'sad=', (e.get('all_probs') or [0]*7)[1] if e.get('all_probs') else '-')"
  done
else
  echo "[SKIP] backend 未启动，跳过 E2E"
fi

echo ""
echo "验收完成 preset=$PRESET"
