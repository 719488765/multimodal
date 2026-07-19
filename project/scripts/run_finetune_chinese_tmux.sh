#!/usr/bin/env bash
# 在 tmux  detached 会话中运行中文微调（SSH 断线不中断）
set -euo pipefail

SESSION="${TMUX_SESSION:-finetune_chinese}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p logs_accuracy_seq
LOG="logs_accuracy_seq/finetune_chinese_tmux_$(date +%Y%m%d_%H%M%S).log"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "ERROR: tmux 会话 '$SESSION' 已存在。" >&2
  echo "  查看: tmux attach -t $SESSION" >&2
  echo "  结束: tmux kill-session -t $SESSION" >&2
  exit 1
fi

RUN_CMD="cd '$ROOT' && export HF_HUB_OFFLINE=$HF_HUB_OFFLINE TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE && ./scripts/finetune_agent_chinese.sh 2>&1 | tee -a '$LOG'"

tmux new-session -d -s "$SESSION" -x 200 -y 50 bash -lc "$RUN_CMD"

sleep 2
# /proc scan only — never pgrep/ps (they hang and fork-storm on this host).
TRAIN_PID="$(python3 - <<'PY'
import os, re
pat = re.compile(r"train\.py.*ap2_M1_chinese_text_agent")
for pid in os.listdir("/proc"):
    if not pid.isdigit():
        continue
    try:
        raw = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        continue
    if pat.search(raw):
        print(pid)
        break
PY
)"

echo "==> tmux 会话: $SESSION"
echo "==> 日志文件: $LOG"
echo "==> 训练 PID: ${TRAIN_PID:-（启动中，稍后用 tmux/日志查看）}"
echo "$LOG" > logs_accuracy_seq/finetune_chinese.latest.log
[[ -n "$TRAIN_PID" ]] && echo "$TRAIN_PID" > logs_accuracy_seq/finetune_chinese.latest.pid

echo ""
echo "常用命令:"
echo "  tmux attach -t $SESSION          # 进入会话（Ctrl+B 再按 D 可脱离）"
echo "  tail -f $LOG"
echo "  tmux kill-session -t $SESSION    # 停止训练（慎用）"
