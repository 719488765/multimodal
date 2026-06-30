#!/usr/bin/env bash
# 无 sudo：Cloudflare Quick Tunnel 暴露 8000（HTTPS，无需 root）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${HOME}/bin/cloudflared"
SESSION="${TMUX_SESSION:-cf_tunnel}"
TARGET="${1:-http://127.0.0.1:8000}"
LOG="${HOME}/cloudflared_tunnel.log"

# 先安装 cloudflared（不依赖 backend），再检查 8000
if [[ ! -x "$BIN" ]]; then
  echo "==> 未找到 cloudflared，开始安装到 $BIN"
  "$ROOT/scripts/install_cloudflared.sh" || exit 1
fi

if ! curl -sf --max-time 8 "${TARGET%/}/api/v1/health" >/dev/null 2>&1; then
  echo "ERROR: ${TARGET} 未响应，请先启动 backend:" >&2
  echo "  cd $ROOT && FORCE_RESTART=1 ./scripts/start_demo.sh" >&2
  echo "  确认健康: curl -s http://127.0.0.1:8000/api/v1/health" >&2
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux 会话 $SESSION 已存在。"
  URL="$(grep -oE 'https://[a-zA-Z0-9-]+\.trycloudflare\.com' "$LOG" 2>/dev/null | tail -1 || true)"
  [[ -n "$URL" ]] && echo "==> 浏览器打开: $URL"
  echo "  tmux attach -t $SESSION"
  exit 0
fi

echo "==> 启动 Quick Tunnel -> $TARGET"
tmux new-session -d -s "$SESSION" \
  "$BIN tunnel --url $TARGET 2>&1 | tee -a $LOG"

sleep 8
URL="$(grep -oE 'https://[a-zA-Z0-9-]+\.trycloudflare\.com' "$LOG" 2>/dev/null | tail -1 || true)"
echo "==> tmux: $SESSION | 日志: $LOG"
if [[ -n "$URL" ]]; then
  echo "==> 浏览器打开: $URL"
  echo "    然后执行: cd $ROOT && ./scripts/build_production.sh && FORCE_RESTART=1 ./scripts/start_demo.sh"
else
  echo "==> URL 生成中: tail -f $LOG  或  tmux attach -t $SESSION"
fi
