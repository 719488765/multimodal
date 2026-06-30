#!/usr/bin/env bash
# 微调完成后切换 emotion-agent 到 agent_chinese preset
set -euo pipefail

ENV_FILE="${2:-../emotion-agent/backend/.env}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CKPT="${1:-}"
if [[ -z "$CKPT" ]]; then
  CKPT="$(ls -t checkpoints_accuracy_seq/AP2_M1_chinese_text_agent_*/checkpoint_finetune_best_f1.pth 2>/dev/null | head -1 || true)"
fi
if [[ -z "$CKPT" && -f checkpoints_accuracy_seq/AP2_M1_chinese_text_agent/checkpoint_finetune_best_f1.pth ]]; then
  CKPT="checkpoints_accuracy_seq/AP2_M1_chinese_text_agent/checkpoint_finetune_best_f1.pth"
fi

if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: checkpoint 不存在" >&2
  if tmux has-session -t finetune_chinese 2>/dev/null; then
    echo "检测到 tmux 会话 finetune_chinese 仍在运行，微调尚未产出 checkpoint_finetune_best_f1.pth。" >&2
    echo "请等待训练完成（或至少完成首个验证 epoch）后再执行本脚本。" >&2
    echo "  tmux attach -t finetune_chinese" >&2
    LATEST_LOG="$(ls -t logs_accuracy_seq/finetune_chinese*.log 2>/dev/null | head -1 || true)"
    [[ -n "$LATEST_LOG" ]] && echo "  tail -f $LATEST_LOG" >&2
  else
    echo "请先运行: ./scripts/finetune_agent_chinese.sh" >&2
    echo "或指定已有 checkpoint: $0 checkpoints_accuracy_seq/.../checkpoint_finetune_best_f1.pth" >&2
  fi
  exit 1
fi

# 链到 preset 固定路径，便于 config.py 加载
PRESET_DIR="checkpoints_accuracy_seq/AP2_M1_chinese_text_agent"
mkdir -p "$PRESET_DIR"
ln -sf "$(readlink -f "$CKPT")" "$PRESET_DIR/checkpoint_finetune_best_f1.pth"
echo "Linked $PRESET_DIR/checkpoint_finetune_best_f1.pth -> $CKPT"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: .env 不存在: $ENV_FILE" >&2
  exit 1
fi

python3 - <<'PY' "$ENV_FILE"
import re, sys
path = sys.argv[1]
text = open(path, encoding="utf-8").read()
text = re.sub(r"^MODEL_CHECKPOINT_PRESET=.*$", "MODEL_CHECKPOINT_PRESET=agent_chinese", text, flags=re.M)
if "MODEL_CHECKPOINT_PRESET=agent_chinese" not in text:
    text += "\nMODEL_CHECKPOINT_PRESET=agent_chinese\n"
text = re.sub(r"^HF_HUB_OFFLINE=.*$", "HF_HUB_OFFLINE=0", text, flags=re.M)
text = re.sub(r"^TRANSFORMERS_OFFLINE=.*$", "TRANSFORMERS_OFFLINE=0", text, flags=re.M)
open(path, "w", encoding="utf-8").write(text)
print("Updated", path, "-> MODEL_CHECKPOINT_PRESET=agent_chinese")
PY

echo "请重启后端: cd ../emotion-agent && FORCE_RESTART=1 ./scripts/start_demo.sh"
