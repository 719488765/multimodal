#!/usr/bin/env bash
# SDAVT R4 单 GPU 队列 worker（experiment_queue.json）
set -euo pipefail

GPU_ID="${1:-0}"
QUEUE_FILE="${2:-outputs_sdavt_v3_r4/experiment_queue.json}"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"

# shellcheck disable=SC1091
source "$PROJECT_DIR/scripts/r4_env.sh"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
PY="${R4_PYTHON:-python3}"

_validate_p4_job() {
  local job_id="$1"
  local dataset="$2"
  if [[ "$job_id" != R4_A_* ]]; then
    return 0
  fi
  if "$PY" scripts/validate_p4_job_metrics.py "$job_id" --dataset "$dataset"; then
    echo "[R4 worker GPU$GPU_ID] $job_id metrics OK (non-collapse)"
  else
    echo "[R4 worker GPU$GPU_ID] WARNING: $job_id may have collapsed metrics"
    return 1
  fi
}

_run_train() {
  local cfg="$1"
  local job_id="$2"
  local extra=()
  if [[ "$job_id" == R4_A_* ]]; then
    extra+=(--replace_log_dir)
  fi
  "$PY" scripts/train.py --config "$cfg" --mode pretrain "${extra[@]}"
}

_mark() {
  "$PY" - "$QUEUE_FILE" "$1" "$2" "${3:-}" <<'PY'
import json, sys
from datetime import datetime
from pathlib import Path

qpath, job_id, status, extra = sys.argv[1:5]
p = Path(qpath)
q = json.loads(p.read_text(encoding="utf-8"))
for j in q["jobs"]:
    if j["id"] == job_id:
        j["status"] = status
        j["updated_at"] = datetime.now().isoformat(timespec="seconds")
        if extra:
            j["note"] = extra
        break
else:
    raise SystemExit(f"job not found: {job_id}")
p.write_text(json.dumps(q, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

_gate_ok() {
  "$PY" - "$QUEUE_FILE" "$1" <<'PY'
import json, sys
from pathlib import Path

qpath, job_id = sys.argv[1:3]
q = json.loads(Path(qpath).read_text(encoding="utf-8"))
job = next(j for j in q["jobs"] if j["id"] == job_id)
dep = job.get("depends_on")
if not dep:
    print("ok")
    raise SystemExit(0)
status_dir = Path("outputs_sdavt_v3_r4/status")
marker = status_dir / f"{dep}.json"
if marker.is_file() or (status_dir / dep).is_file():
    print("ok")
    raise SystemExit(0)
# also accept if dependency is a job id that is done
for j in q["jobs"]:
    if j["id"] == dep and j.get("status") == "done":
        print("ok")
        raise SystemExit(0)
print("wait")
PY
}

_next_job() {
  "$PY" - "$QUEUE_FILE" "$GPU_ID" <<'PY'
import json, sys
from pathlib import Path

qpath, gpu = sys.argv[1], int(sys.argv[2])
q = json.loads(Path(qpath).read_text(encoding="utf-8"))
status_dir = Path("outputs_sdavt_v3_r4/status")
MOSEI_AUDIO_RERUN = {"R4_A_O_AT", "R4_A_O_AVT", "R4_A_O_AV", "R4_A_O_A"}

def gate_ok(job):
    dep = job.get("depends_on")
    if not dep:
        return True
    if (status_dir / f"{dep}.json").is_file() or (status_dir / dep).is_file():
        return True
    for j in q["jobs"]:
        if j["id"] == dep and j.get("status") == "done":
            return True
    return False

pending = [j for j in q["jobs"] if j.get("status") == "pending" and gate_ok(j) and j["id"] not in MOSEI_AUDIO_RERUN]
if not pending:
    print("")
    raise SystemExit(0)
pending.sort(key=lambda j: (
    0 if int(j.get("gpu_hint", gpu)) == gpu else 1,
    j.get("priority", 9),
    j.get("id", ""),
))
print(pending[0]["id"])
PY
}

echo "[R4 worker GPU$GPU_ID] queue=$QUEUE_FILE"

while true; do
  JOB_ID="$(_next_job || true)"
  if [[ -z "$JOB_ID" ]]; then
    echo "[R4 worker GPU$GPU_ID] no runnable pending jobs — exit"
    break
  fi

  CONFIG="$("$PY" - <<PY
import json
from pathlib import Path
q = json.loads(Path("$QUEUE_FILE").read_text(encoding="utf-8"))
for j in q["jobs"]:
    if j["id"] == "$JOB_ID":
        print(j["config"])
        break
PY
)"

  DATASET="$("$PY" - <<PY
import json
from pathlib import Path
q = json.loads(Path("$QUEUE_FILE").read_text(encoding="utf-8"))
for j in q["jobs"]:
    if j["id"] == "$JOB_ID":
        print(j.get("dataset", ""))
        break
PY
)"

  echo "[R4 worker GPU$GPU_ID] >>> $JOB_ID config=$CONFIG"
  _mark "$JOB_ID" "running" "gpu$GPU_ID"

  if _run_train "$CONFIG" "$JOB_ID"; then
    if _validate_p4_job "$JOB_ID" "$DATASET"; then
      _mark "$JOB_ID" "done" "gpu$GPU_ID"
      echo "[R4 worker GPU$GPU_ID] <<< $JOB_ID done"
    else
      _mark "$JOB_ID" "failed" "gpu$GPU_ID collapse_check"
      echo "[R4 worker GPU$GPU_ID] <<< $JOB_ID FAILED (collapse check)"
    fi
  else
    _mark "$JOB_ID" "failed" "gpu$GPU_ID"
    echo "[R4 worker GPU$GPU_ID] <<< $JOB_ID FAILED"
  fi
done

echo "[R4 worker GPU$GPU_ID] finished"
