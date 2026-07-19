#!/usr/bin/env bash
# SDAVT v3 R4 论文轨 TensorBoard（默认 port 6008，与旧轨 6007 隔离）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${1:-6008}"
LOGDIR="${LOGDIR:-$ROOT/logs_sdavt_v3_r4}"

# Prefer project conda env when tensorboard is not on default PATH.
if command -v tensorboard >/dev/null 2>&1; then
  TB_BIN="$(command -v tensorboard)"
elif [[ -x "${HOME}/.conda/envs/myenv310/bin/tensorboard" ]]; then
  TB_BIN="${HOME}/.conda/envs/myenv310/bin/tensorboard"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/tensorboard" ]]; then
  TB_BIN="${CONDA_PREFIX}/bin/tensorboard"
else
  echo "ERROR: tensorboard not found. Activate myenv310 or install tensorboard." >&2
  exit 1
fi

if [[ ! -d "$LOGDIR" ]]; then
  echo "ERROR: logdir not found: $LOGDIR" >&2
  exit 1
fi

echo "R4 TensorBoard logdir: $LOGDIR (port $PORT)"
echo "TensorBoard binary: $TB_BIN"
echo "URL: http://127.0.0.1:${PORT}/"
echo "Archived (excluded from TB): ${ROOT}/logs_sdavt_v3_r4_archived/"
echo "Prune retries: python scripts/prune_r4_tensorboard_logs.py [--apply]"
echo "Deprecated (do NOT use for paper): logs_sdavt_v3 logs_accuracy_seq"
exec "$TB_BIN" --logdir "$LOGDIR" --port "$PORT" --bind_all
