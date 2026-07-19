#!/usr/bin/env bash
# 单端口演示：构建前端 + 启动后端，浏览器只需访问 http://127.0.0.1:8000
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FRONTEND="$ROOT/frontend"
BACKEND="$ROOT/backend"
FORCE_RESTART="${FORCE_RESTART:-0}"
SAFE_FREE_PORT="$ROOT/scripts/safe_free_port.sh"

port_in_use() {
  ss -tlnp 2>/dev/null | grep -q ':8000 '
}

free_port_8000() {
  # Never use fuser/ps — they hang and can fork-storm this host.
  if [[ -x "$SAFE_FREE_PORT" ]]; then
    bash "$SAFE_FREE_PORT" 8000 || true
  else
    echo "WARN: missing $SAFE_FREE_PORT; skip port kill"
  fi
}

health_ok() {
  curl -sf http://127.0.0.1:8000/api/v1/health >/dev/null 2>&1
}

echo "==> 1/3 构建前端（同源 API，无需 5173 代理）"
cd "$FRONTEND"
if [[ ! -d node_modules ]]; then
  npm install
fi
npm run build

echo "==> 2/3 检查 ASR (9010)"
if ! curl -sf http://127.0.0.1:9010/health >/dev/null 2>&1; then
  echo "WARN: ASR 未运行。另开终端: cd $ROOT/asr-local && ./start_server.sh"
fi

if ss -tlnp 2>/dev/null | grep -q ':5173 '; then
  echo "WARN: 5173 上仍有 Vite dev。大 POST 经代理易失败，请关闭 npm run dev，仅用 http://127.0.0.1:8000"
fi

echo "==> 3/3 启动后端 (8000)"
if port_in_use; then
  if [[ "$FORCE_RESTART" == "1" ]]; then
    echo "释放 8000 端口（FORCE_RESTART=1，无 fuser/ps）..."
    free_port_8000
    sleep 2
  elif health_ok; then
    echo ""
    echo "=============================================="
    echo "  8000 已在运行且健康，无需重复启动。"
    echo "  浏览器打开:  http://127.0.0.1:8000"
    echo "  若要强制重启: FORCE_RESTART=1 ./scripts/start_demo.sh"
    echo "=============================================="
    curl -s http://127.0.0.1:8000/api/v1/health | python3 -m json.tool 2>/dev/null | head -12 || true
    exit 0
  else
    echo "8000 被占用但 health 失败，正在释放端口（无 fuser/ps）..."
    free_port_8000
    sleep 2
  fi
fi

echo ""
echo "=============================================="
echo "  请在浏览器打开:  http://127.0.0.1:8000"
echo "  Cursor: Ports 面板确认 8000 已转发"
echo "=============================================="
echo ""

cd "$BACKEND"
exec ./start_server.sh
