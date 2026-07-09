#!/usr/bin/env bash
# Rerun collapsed MOSEI P4 jobs (AT/AVT/AV/A) after selective temporal_encoder fix.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"

# shellcheck disable=SC1091
source "$PROJECT_DIR/scripts/r4_env.sh"
PY="${R4_PYTHON:-python3}"

ARCH="outputs_sdavt_v3_r4/archived/p4_mosei_audio_collapse_20260701"
mkdir -p "$ARCH"

for job in R4_A_O_AT R4_A_O_AVT R4_A_O_AV R4_A_O_A; do
  for base in logs_sdavt_v3_r4 checkpoints_sdavt_v3_r4; do
    d="$base/SDAVT_R4_${job}"
    if [[ -d "$d" ]]; then
      dest="$ARCH/$(basename "$d")"
      rm -rf "$dest"
      mv "$d" "$ARCH/" && echo "archived $d"
    fi
  done
done

"$PY" scripts/reset_r4_queue_jobs.py R4_A_O_AT R4_A_O_AVT R4_A_O_AV R4_A_O_A --phase p4_modal

export CUDA_VISIBLE_DEVICES=0
for job in R4_A_O_AT R4_A_O_AVT R4_A_O_AV R4_A_O_A; do
  cfg="$("$PY" - <<PY
import json
q=json.load(open("outputs_sdavt_v3_r4/experiment_queue.json",encoding="utf-8"))
for j in q["jobs"]:
    if j["id"]=="$job":
        print(j["config"]); break
PY
)"
  echo ">>> $job $cfg"
  "$PY" scripts/train.py --config "$cfg" --mode pretrain --replace_log_dir
  "$PY" scripts/validate_p4_job_metrics.py "$job" --dataset mosei || {
    echo "FAIL: $job collapse check"
    exit 1
  }
done

"$PY" scripts/validate_p4_job_metrics.py --dataset mosei --strict || {
  echo "FAIL: MOSEI strict validation"
  exit 1
}

bash scripts/start_sdavt_r4.sh report
echo "[OK] MOSEI audio P4 rerun complete"
