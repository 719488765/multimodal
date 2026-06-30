#!/usr/bin/env bash
# Author: AI
# Date: 2026-04-11
# Description: 启动 TensorBoard，指向准确率优化实验序列日志目录 logs_accuracy_seq

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${1:-6006}"

cd "$PROJECT_DIR"
echo "[INFO] logdir=$PROJECT_DIR/logs_accuracy_seq port=$PORT"
exec tensorboard --logdir "$PROJECT_DIR/logs_accuracy_seq" --port "$PORT" --bind_all
