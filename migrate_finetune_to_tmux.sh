#!/usr/bin/env bash
# 将正在运行的微调迁移到 tmux（若无 epoch 进度则安全重启）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PROJECT="$ROOT/project"
SESSION="${TMUX_SESSION:-finetune_chinese}"

cd "$PROJECT"

PIDS=($(pgrep -f 'train.py.*ap2_M1_chinese_text_agent' || true))

if [[ ${#PIDS[@]} -eq 0 ]]; then
  echo "未发现运行中的 chinese 微调，直接启动 tmux..."
  exec "$PROJECT/scripts/run_finetune_chinese_tmux.sh"
fi

if [[ ${#PIDS[@]} -gt 1 ]]; then
  echo "WARN: 发现 ${#PIDS[@]} 个 train.py 进程: ${PIDS[*]}"
  echo "      将停止全部后在 tmux 中重启一个。"
fi

# 检查是否已有 epoch 记录
EPOCH_LINES=0
for m in logs_accuracy_seq/AP2_M1_chinese_text_agent_*/metrics.csv; do
  [[ -f "$m" ]] || continue
  n=$(($(wc -l < "$m") - 1))
  if [[ "$n" -gt 0 ]]; then
    EPOCH_LINES=$n
    METRICS_FILE="$m"
    break
  fi
done

if [[ "$EPOCH_LINES" -gt 0 ]]; then
  echo "ERROR: 已在训练（$METRICS_FILE 有 $EPOCH_LINES 行 epoch），不宜自动 kill。" >&2
  echo "请手动: tmux attach 或安装 reptyr 迁移进程。" >&2
  exit 1
fi

echo "==> 当前微调尚未进入 Epoch（仅数据加载阶段），安全迁移到 tmux..."
for pid in "${PIDS[@]}"; do
  echo "    停止 PID $pid"
  kill "$pid" 2>/dev/null || true
done
sleep 3
for pid in "${PIDS[@]}"; do
  kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
done

if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION"
fi

exec "$PROJECT/scripts/run_finetune_chinese_tmux.sh"
