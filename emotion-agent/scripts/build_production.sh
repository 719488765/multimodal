#!/usr/bin/env bash
# 生产构建：server 模式前端 + 可选重启后端
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/frontend"

export VITE_API_BASE_URL=
export VITE_DEPLOY_MODE=server

echo "==> npm run build (VITE_DEPLOY_MODE=server)"
npm run build

echo "==> dist ready at frontend/dist"
echo "    启动: cd $ROOT && ./scripts/start_demo.sh"
echo "    或:   FORCE_RESTART=1 ./scripts/start_demo.sh"
