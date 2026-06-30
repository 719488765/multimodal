#!/usr/bin/env bash
# 切换 emotion-agent 部署权重 preset
# 用法: ./scripts/apply_deploy_preset.sh meld_only|ap2_m1|mosei_only|agent_chinese
set -euo pipefail

PRESET="${1:-meld_only}"
ENV_FILE="${2:-../emotion-agent/backend/.env}"

case "$PRESET" in
  meld_only|mosei_only|ap2_m1|ap4_w005|agent_chinese|sdavt_meld_v3_r4) ;;
  *)
    echo "ERROR: 未知 preset: $PRESET" >&2
    echo "可选: meld_only mosei_only ap2_m1 ap4_w005 agent_chinese sdavt_meld_v3_r4" >&2
    exit 1
    ;;
esac

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: .env 不存在: $ENV_FILE" >&2
  exit 1
fi

python3 - <<'PY' "$ENV_FILE" "$PRESET"
import re, sys
path, preset = sys.argv[1], sys.argv[2]
text = open(path, encoding="utf-8").read()
line = f"MODEL_CHECKPOINT_PRESET={preset}"
if re.search(r"^MODEL_CHECKPOINT_PRESET=", text, flags=re.M):
    text = re.sub(r"^MODEL_CHECKPOINT_PRESET=.*$", line, text, flags=re.M)
else:
    text += "\n" + line + "\n"
open(path, "w", encoding="utf-8").write(text)
print(f"Updated {path} -> {line}")
PY

echo ""
echo "Preset: $PRESET"
echo "  meld_only  — MELD 单域 mp4，与在线 ResNet 抽帧一致（推荐 Agent）"
echo "  mosei_only — MOSEI 单域（训练多为 npy 特征，在线有域差，实验用）"
echo "  ap2_m1     — 三混合 Best F1≈0.56"
echo "  agent_chinese — 中文 BERT 三混合微调"
echo "  sdavt_meld_v3_r4 — R4 P3 冠军 M3_M3_uniform F1≈0.611"
echo ""
echo "请重启: cd ../emotion-agent && FORCE_RESTART=1 ./scripts/start_demo.sh"
echo "浏览器 Ctrl+Shift+R 强刷"
