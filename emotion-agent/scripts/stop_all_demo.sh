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

for port in 8000 9010; do
  if ss -tlnp 2>/dev/null | grep -q ":${port} "; then
    echo "释放端口 ${port}..."
    fuser -k "${port}/tcp" 2>/dev/null || true
  fi
done

echo "完成。Ollama (11434) 未停止（通常为系统服务）。"
