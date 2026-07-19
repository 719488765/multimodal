#!/usr/bin/env bash
# Start R4 TensorBoard in tmux if not already listening on port 6008.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${1:-6008}"
SESSION="${TB_TMUX_SESSION:-sdavt_r4_tensorboard}"

if curl -sf --connect-timeout 2 "http://127.0.0.1:${PORT}/" >/dev/null 2>&1; then
  echo "[OK] TensorBoard already up: http://127.0.0.1:${PORT}/"
  exit 0
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION"
fi

tmux new-session -d -s "$SESSION" \
  "cd '$ROOT' && exec bash scripts/tensorboard_sdavt_r4.sh '$PORT'"

for _ in $(seq 1 15); do
  if curl -sf --connect-timeout 2 "http://127.0.0.1:${PORT}/" >/dev/null 2>&1; then
    echo "[OK] TensorBoard started: http://127.0.0.1:${PORT}/"
    echo "     tmux attach -t $SESSION"
    exit 0
  fi
  sleep 1
done

echo "ERROR: TensorBoard failed to start on port $PORT" >&2
echo "Check: tmux attach -t $SESSION" >&2
exit 1
