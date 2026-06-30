#!/usr/bin/env bash
# Cursor 远程开发：tunnel 模式构建（127.0.0.1:8000 小包单次 multipart）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/frontend"
export VITE_API_BASE_URL=
export VITE_DEPLOY_MODE=tunnel
echo "==> npm run build (VITE_DEPLOY_MODE=tunnel, 适配 Cursor 端口转发)"
npm run build
echo "==> 完成后: FORCE_RESTART=1 ../scripts/start_demo.sh"
