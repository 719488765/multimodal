#!/usr/bin/env bash
# 在远程服务器上启动静态 HTTP，供本地通过 SSH/Cursor 端口转发访问架构图 HTML。
#
# 本地浏览器打开（转发 8765 后）:
#   http://127.0.0.1:8765/system_architecture_figure.html   ← 一张总图（Figure 1）
#   http://127.0.0.1:8765/thesis/index.html               ← 分图说明与预览
#
# Cursor Remote SSH: 终端运行本脚本 → 底部 Ports 面板 Forward 8765 → 点击 Open in Browser
#
# 本地终端 SSH 隧道:
#   ssh -L 8765:127.0.0.1:8765 USER@REMOTE_HOST
#   然后浏览器打开 http://127.0.0.1:8765/system_architecture_figure.html

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FIG_DIR="${PROJECT_ROOT}/docs/figures"
HTML="${FIG_DIR}/system_architecture_figure.html"
PORT="${ARCH_FIGURE_PORT:-8765}"

if [[ ! -f "${HTML}" ]]; then
  echo "ERROR: missing ${HTML}" >&2
  exit 1
fi

if command -v ss >/dev/null 2>&1; then
  if ss -tln | grep -q ":${PORT} "; then
    echo "Port ${PORT} already in use. Kill the old process or set ARCH_FIGURE_PORT=8766"
    ss -tlnp | grep ":${PORT} " || true
    exit 1
  fi
fi

echo "Serving: ${HTML}"
echo "Directory: ${FIG_DIR}"
echo "URL (after port forward): http://127.0.0.1:${PORT}/system_architecture_figure.html"
echo "Thesis index:             http://127.0.0.1:${PORT}/thesis/index.html"
echo "Press Ctrl+C to stop."
echo

cd "${FIG_DIR}"
exec python3 -m http.server "${PORT}" --bind 127.0.0.1
