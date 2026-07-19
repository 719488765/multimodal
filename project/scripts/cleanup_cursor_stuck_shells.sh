#!/usr/bin/env bash
# Kill stuck Cursor agent sandbox shells (fuser/pgrep/ps hangs) without touching services.
# Does NOT use ps/pgrep/fuser itself — only /proc + kill.
set -uo pipefail

KEEP_PORTS=(6008 8000 9010)
MY_UID="$(id -u)"
MY_PID="$$"
MY_PPID="$(awk '{print $4}' /proc/$$/stat 2>/dev/null || echo 0)"

kill_if_match() {
  local pid="$1"
  local cmd="$2"
  if [[ "$pid" == "$MY_PID" || "$pid" == "$MY_PPID" ]]; then
    return 0
  fi
  # Never kill cursor-server or tmux server.
  if [[ "$cmd" == *cursor-server* || "$cmd" == *tmux* ]]; then
    return 0
  fi
  if [[ "$cmd" == *"/bin/bash -O extglob"* && "$cmd" == *dump_bash_state* ]]; then
    kill -9 "$pid" 2>/dev/null && echo "[kill] sandbox bash pid=$pid"
    return 0
  fi
  # Stuck process scanners — these hang on this host and fork-storm CPU.
  if [[ "$cmd" == *" fuser"* || "$cmd" == fuser* \
     || "$cmd" == pkill* || "$cmd" == *" pkill "* \
     || "$cmd" == pgrep* || "$cmd" == *" pgrep "* \
     || "$cmd" == "ps "* || "$cmd" == */ps\ * || "$cmd" == "ps" ]]; then
    kill -9 "$pid" 2>/dev/null && echo "[kill] stuck diag pid=$pid cmd=${cmd:0:80}"
    return 0
  fi
  if [[ "$cmd" == *"os.listdir('/proc')"* && "$cmd" != *cursor-server* ]]; then
    kill -9 "$pid" 2>/dev/null && echo "[kill] stuck proc-scan pid=$pid"
  fi
}

for proc in /proc/[0-9]*; do
  pid="${proc##*/}"
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  if [[ ! -r "$proc/status" ]]; then
    continue
  fi
  uid="$(awk '/^Uid:/{print $2}' "$proc/status" 2>/dev/null || true)"
  [[ "$uid" == "$MY_UID" ]] || continue
  state="$(awk '/^State:/{print $2}' "$proc/status" 2>/dev/null || true)"
  [[ "$state" == "D" || "$state" == "Z" ]] && continue
  cmd="$(timeout 0.2 tr '\0' ' ' < "$proc/cmdline" 2>/dev/null || true)"
  [[ -n "$cmd" ]] || continue
  kill_if_match "$pid" "$cmd"
done

# Stop obsolete R4 watch tmux sessions (GPU line closed; loops waste CPU).
for sess in r4_progress_watch r4_training_completion_watch watch_p3_milestones \
            watch_r4_p2_es p2_es_retrain_watch; do
  if tmux has-session -t "$sess" 2>/dev/null; then
    tmux kill-session -t "$sess" 2>/dev/null && echo "[kill] tmux $sess"
  fi
done

echo "[check] services:"
for port in "${KEEP_PORTS[@]}"; do
  code="$(curl -sf -o /dev/null -w '%{http_code}' --connect-timeout 2 "http://127.0.0.1:${port}/" 2>/dev/null || echo fail)"
  echo "  port ${port}: ${code}"
done
echo "[done] remaining tmux:"; tmux ls 2>/dev/null || true
