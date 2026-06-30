#!/usr/bin/env bash
# Start emotion-agent backend with the training conda env (uvicorn is not on system PATH).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-/home/lizhichun_24/.conda/envs/myenv310/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Python not found: $PYTHON" >&2
  echo "Set PYTHON= to your conda env python (e.g. myenv310)." >&2
  exit 1
fi

if ! "$PYTHON" -c "import uvicorn" 2>/dev/null; then
  echo "Installing backend dependencies..."
  "$PYTHON" -m pip install -r requirements.txt
fi

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

echo "Using: $PYTHON"
echo "Starting http://${HOST}:${PORT} (MODEL_PROVIDER from .env)..."
exec "$PYTHON" -m uvicorn app.main:app --host "$HOST" --port "$PORT" --timeout-keep-alive 600
