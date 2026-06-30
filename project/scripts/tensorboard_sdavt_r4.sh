#!/usr/bin/env bash
# SDAVT v3 R4 论文轨 TensorBoard（默认 port 6008，与旧轨 6007 隔离）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${1:-6008}"
LOGDIR="${LOGDIR:-$ROOT/logs_sdavt_v3_r4}"

echo "R4 TensorBoard logdir: $LOGDIR (port $PORT)"
echo "Deprecated (do NOT use for paper): logs_sdavt_v3 logs_accuracy_seq"
exec tensorboard --logdir "$LOGDIR" --port "$PORT" --bind_all
