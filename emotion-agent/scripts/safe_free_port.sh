#!/usr/bin/env bash
# Free a TCP listen port WITHOUT fuser/ps/pgrep (those hang and fork-storm on this host).
# Usage: safe_free_port.sh <port> [...]
set -uo pipefail

safe_free_port() {
  local port="$1"
  local pids=""
  local pid

  if command -v ss >/dev/null 2>&1; then
    # ss prints users:(("cmd",pid=123,fd=4))
    pids="$(
      timeout 2 ss -H -tlnp "sport = :${port}" 2>/dev/null \
        | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' \
        | sort -u
    )" || true
  fi

  if [[ -z "${pids}" ]]; then
    # Fallback: parse /proc/net/tcp + /proc/*/fd (no ps/fuser).
    pids="$(
      timeout 3 python3 - "$port" <<'PY' 2>/dev/null || true
import os, sys
port = int(sys.argv[1])
want = f"{port:04X}"
inodes = set()
for path in ("/proc/net/tcp", "/proc/net/tcp6"):
    try:
        with open(path) as f:
            next(f)
            for line in f:
                parts = line.split()
                if len(parts) < 10:
                    continue
                local = parts[1]
                if local.endswith(":" + want) and parts[3] == "0A":  # LISTEN
                    inodes.add(parts[9])
    except OSError:
        pass
if not inodes:
    sys.exit(0)
found = set()
for pid in os.listdir("/proc"):
    if not pid.isdigit():
        continue
    fd_dir = f"/proc/{pid}/fd"
    try:
        for fd in os.listdir(fd_dir):
            try:
                target = os.readlink(f"{fd_dir}/{fd}")
            except OSError:
                continue
            if target.startswith("socket:[") and target[8:-1] in inodes:
                found.add(pid)
                break
    except OSError:
        continue
print("\n".join(sorted(found)))
PY
    )"
  fi

  if [[ -z "${pids}" ]]; then
    echo "[safe_free_port] port ${port}: no listener"
    return 0
  fi

  echo "[safe_free_port] port ${port}: TERM pids=${pids//$'\n'/ }"
  for pid in $pids; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  sleep 1
  for pid in $pids; do
    if [[ -d "/proc/${pid}" ]]; then
      echo "[safe_free_port] port ${port}: KILL pid=${pid}"
      kill -KILL "$pid" 2>/dev/null || true
    fi
  done
}

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <port> [port...]" >&2
  exit 2
fi

for p in "$@"; do
  safe_free_port "$p"
done
