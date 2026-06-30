#!/usr/bin/env bash
# 后台启动 bert-base-chinese 微调（自动 cd 到 project/ 并创建日志目录）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p logs_accuracy_seq
LOG="logs_accuracy_seq/finetune_chinese_$(date +%Y%m%d_%H%M%S).log"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"

echo "==> 工作目录: $ROOT"
echo "==> 日志文件: $LOG"
echo "==> 查看进度: tail -f $LOG"

nohup env HF_HUB_OFFLINE="$HF_HUB_OFFLINE" TRANSFORMERS_OFFLINE="$TRANSFORMERS_OFFLINE" \
  ./scripts/finetune_agent_chinese.sh >> "$LOG" 2>&1 &

PID=$!
echo "==> 已启动 PID=$PID"
echo "$PID" > logs_accuracy_seq/finetune_chinese.latest.pid
echo "$LOG" > logs_accuracy_seq/finetune_chinese.latest.log

sleep 2
if kill -0 "$PID" 2>/dev/null; then
  echo "==> 训练进程运行中"
  tail -n 15 "$LOG"
else
  echo "ERROR: 进程已退出，请查看日志:" >&2
  tail -n 30 "$LOG" >&2
  exit 1
fi
