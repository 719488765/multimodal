#!/usr/bin/env bash
# 单 GPU 消融队列 worker：从 ablation_queue.json 取 pending 任务并训练
# 由 start_sdavt_ablation.sh 启动，勿手动重复开多个同 GPU worker

set -euo pipefail

GPU_ID="${1:-0}"
QUEUE_FILE="${2:-outputs_sdavt_v3/ablation_queue.json}"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"

export CUDA_VISIBLE_DEVICES="$GPU_ID"

_run_train() {
  local cfg="$1"
  python3 scripts/train.py --config "$cfg" --mode pretrain
}

_mark() {
  python3 - "$QUEUE_FILE" "$1" "$2" "${3:-}" <<'PY'
import json, sys
from datetime import datetime
from pathlib import Path

qpath, job_id, status, extra = sys.argv[1:5]
p = Path(qpath)
q = json.loads(p.read_text())
for j in q["jobs"]:
    if j["id"] == job_id:
        j["status"] = status
        j["updated_at"] = datetime.now().isoformat(timespec="seconds")
        if extra:
            j["note"] = extra
        break
else:
    raise SystemExit(f"job not found: {job_id}")
p.write_text(json.dumps(q, indent=2, ensure_ascii=False) + "\n")
PY
}

_next_job() {
  python3 - "$QUEUE_FILE" "$GPU_ID" <<'PY'
import json, sys
from pathlib import Path

qpath, gpu = sys.argv[1], int(sys.argv[2])
q = json.loads(Path(qpath).read_text())
pending = [j for j in q["jobs"] if j.get("status") == "pending"]
if not pending:
    print("")
    raise SystemExit(0)
# 优先 gpu_hint 匹配，其次任意
pending.sort(key=lambda j: (
    0 if int(j.get("gpu_hint", gpu)) == gpu else 1,
    j.get("priority", 9),
    j.get("id", ""),
))
print(pending[0]["id"])
PY
}

echo "[worker GPU$GPU_ID] queue=$QUEUE_FILE"

while true; do
  JOB_ID="$(_next_job || true)"
  if [[ -z "$JOB_ID" ]]; then
    echo "[worker GPU$GPU_ID] no pending jobs — exit"
    break
  fi

  CONFIG="$(python3 - <<PY
import json
from pathlib import Path
q = json.loads(Path("$QUEUE_FILE").read_text())
for j in q["jobs"]:
    if j["id"] == "$JOB_ID":
        print(j["config"])
        break
PY
)"

  echo "[worker GPU$GPU_ID] >>> $JOB_ID config=$CONFIG"
  _mark "$JOB_ID" "running" "gpu$GPU_ID"

  if _run_train "$CONFIG"; then
    _mark "$JOB_ID" "done" "gpu$GPU_ID"
    echo "[worker GPU$GPU_ID] <<< $JOB_ID done"
  else
    _mark "$JOB_ID" "failed" "gpu$GPU_ID"
    echo "[worker GPU$GPU_ID] <<< $JOB_ID FAILED (see log)"
  fi
done

echo "[worker GPU$GPU_ID] finished"
