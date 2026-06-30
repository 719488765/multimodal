#!/usr/bin/env bash
# SDAVT v3 TensorBoard（默认 port 6007）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${1:-6007}"
LOGDIR="${LOGDIR:-$ROOT/logs_sdavt_v3}"

echo "TensorBoard logdir: $LOGDIR (port $PORT)"
exec tensorboard --logdir "$LOGDIR" --port "$PORT" --bind_all
