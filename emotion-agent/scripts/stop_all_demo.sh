#!/usr/bin/env bash
# 停止 tmux 演示会话并释放 8000 / 9010 端口
set -euo pipefail

SESSION="${TMUX_SESSION:-emotion-demo}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "停止 tmux 会话: $SESSION"
  tmux kill-session -t "$SESSION"
else
  echo "tmux 会话 $SESSION 不存在，跳过"
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SAFE_FREE_PORT="$ROOT/scripts/safe_free_port.sh"
for port in 8000 9010; do
  if ss -tlnp 2>/dev/null | grep -q ":${port} "; then
    echo "释放端口 ${port}（无 fuser/ps）..."
    bash "$SAFE_FREE_PORT" "$port" || true
  fi
done

echo "完成。Ollama (11434) 未停止（通常为系统服务）。"
