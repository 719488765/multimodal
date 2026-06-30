#!/usr/bin/env bash
# 停止 Emotion Agent 全栈：tmux 会话 + 8000/9010 端口 + cloudflared
set -euo pipefail

SESSION="${TMUX_SESSION:-emotion-full}"
CF_SESSION="${CF_TMUX_SESSION:-cf_tunnel}"

for s in "$SESSION" emotion-demo "$CF_SESSION" cf_tunnel; do
  if tmux has-session -t "$s" 2>/dev/null; then
    echo "停止 tmux 会话: $s"
    tmux kill-session -t "$s"
  fi
done

for port in 8000 9010; do
  if ss -tlnp 2>/dev/null | grep -q ":${port} "; then
    echo "释放端口 ${port}..."
    fuser -k "${port}/tcp" 2>/dev/null || true
  fi
done

echo ""
echo "完成。Ollama (11434) 未停止（通常为后台服务）。"
echo "重新启动: cd emotion-agent && ./scripts/start_full_stack.sh"
