#!/usr/bin/env bash
# 一键启动演示：Ollama 检查 + ASR(9010) + 后端+前端(8000)，使用 tmux 会话 emotion-demo
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SESSION="${TMUX_SESSION:-emotion-demo}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "ERROR: 未安装 tmux。请按 START_DEMO.txt「二、每次演示启动」手动开 3 个终端。" >&2
  exit 1
fi

echo "==> 0/3 检查 Ollama (11434)"
cd "$ROOT"
chmod +x scripts/start_ollama.sh 2>/dev/null || true
./scripts/start_ollama.sh || echo "WARN: Ollama 未就绪，LLM 将降级为 template"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo ""
  echo "tmux 会话 '$SESSION' 已存在。"
  echo "  进入: tmux attach -t $SESSION"
  echo "  重建: ./scripts/stop_all_demo.sh && ./scripts/start_all_demo.sh"
  exit 0
fi

echo "==> 1/3 创建 tmux 窗口 0 — ASR (9010)"
tmux new-session -d -s "$SESSION" -n asr -c "$ROOT/asr-local"
tmux send-keys -t "$SESSION:asr" "./start_server.sh" Enter

echo "==> 2/3 创建 tmux 窗口 1 — 等待 ASR 就绪"
tmux new-window -t "$SESSION" -n wait -c "$ROOT"
tmux send-keys -t "$SESSION:wait" "for i in \$(seq 1 60); do curl -sf http://127.0.0.1:9010/health >/dev/null && echo 'ASR ready' && break; sleep 2; done" Enter

echo "==> 3/3 创建 tmux 窗口 2 — 后端+前端 (8000)"
tmux new-window -t "$SESSION" -n demo -c "$ROOT"
tmux send-keys -t "$SESSION:demo" "sleep 5 && ./scripts/start_demo.sh" Enter

echo ""
echo "=============================================="
echo "  tmux 会话已启动: $SESSION"
echo "  进入查看日志:    tmux attach -t $SESSION"
echo "  窗口: 0=asr  1=wait  2=demo"
echo "  浏览器打开:      http://127.0.0.1:8000"
echo "  Cursor 转发端口 8000"
echo "=============================================="
echo ""
echo "约 1–2 分钟后自检:"
echo "  curl -s http://127.0.0.1:8000/api/v1/health | python3 -m json.tool"
