#!/usr/bin/env bash
# Isolate and retrain F_C_ES after AVT slot collision (metrics/ckpt MD5 duplicate).
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
ARCH="outputs_sdavt_v3_r4/archived/f_c_es_avt_collision_${TS}"
STATUS_DIR="outputs_sdavt_v3_r4/status"
CFG="config/sdavt_v3_r4/p2_fusion/crema/F_C_ES_emotion_shift.yaml"
AVT_RUN="SDAVT_R4_R4_A_C_AVT"
FCES_RUN="SDAVT_R4_F_C_ES"
GPU="1"

mkdir -p "$ARCH" "$STATUS_DIR"

# Archive polluted F_C_ES slot
for base in logs_sdavt_v3_r4 checkpoints_sdavt_v3_r4; do
  d="$base/$FCES_RUN"
  if [[ -d "$d" ]]; then
    dest="$ARCH/$(basename "$d")"
    rm -rf "$dest"
    mv "$d" "$ARCH/" && echo "archived $d -> $ARCH"
  fi
done

# Pre-flight: ensure yaml points to exclusive slot
python3 - <<'PY'
import yaml, sys
cfg = yaml.safe_load(open("config/sdavt_v3_r4/p2_fusion/crema/F_C_ES_emotion_shift.yaml", encoding="utf-8"))
exp = cfg.get("experiment", {})
assert exp.get("log_run_dir") == "SDAVT_R4_F_C_ES", exp.get("log_run_dir")
assert exp.get("job_id") == "F_C_ES", exp.get("job_id")
assert exp.get("replace_log_dir") is True, "replace_log_dir must be true"
print("[OK] F_C_ES yaml slot config verified")
PY

echo ">>> F_C_ES isolated retrain on GPU${GPU}"
unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES="$GPU"
python3 scripts/train.py --config "$CFG" --mode pretrain --replace_log_dir \
  >> "$STATUS_DIR/retrain_f_c_es_isolated.log" 2>&1 || {
  _log_fail "train failed; attempting backup restore"
  python3 scripts/restore_f_c_es_from_backup.py >> "$STATUS_DIR/retrain_f_c_es_isolated.log" 2>&1
  exit $?
}

# Verify not identical to AVT
python3 - <<PY
import hashlib
from pathlib import Path
log = Path("logs_sdavt_v3_r4")
avt = log / "$AVT_RUN" / "metrics.csv"
fces = log / "$FCES_RUN" / "metrics.csv"
if not fces.is_file():
    raise SystemExit("missing F_C_ES metrics after retrain")
h_avt = hashlib.md5(avt.read_bytes()).hexdigest() if avt.is_file() else None
h_fces = hashlib.md5(fces.read_bytes()).hexdigest()
if h_avt and h_avt == h_fces:
    raise SystemExit(f"F_C_ES metrics still identical to AVT md5={h_avt}")
print(f"[OK] F_C_ES metrics distinct from AVT (md5={h_fces[:12]})")
PY

python3 scripts/finalize_r4_p2_retrain.py F_C_ES --status done --note "isolated_retrain_${TS}"

# Accept F1 >= 0.54
python3 - <<'PY'
import csv
from pathlib import Path
p = Path("logs_sdavt_v3_r4/SDAVT_R4_F_C_ES/metrics.csv")
best = None
with p.open() as f:
    for r in csv.DictReader(f):
        if r.get("phase") == "val" and r.get("f1"):
            v = float(r["f1"])
            best = v if best is None else max(best, v)
if best is None or best < 0.54:
    raise SystemExit(f"F_C_ES best F1 {best} < 0.54")
print(f"[OK] F_C_ES best F1={best:.4f} >= 0.54")
PY

rm -f "$STATUS_DIR/f_c_es_antiof_retrain_done"
touch "$STATUS_DIR/f_c_es_isolated_retrain_done"
echo "[OK] F_C_ES isolated retrain complete"
