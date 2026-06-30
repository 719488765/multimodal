#!/usr/bin/env bash
# Verify AP3 two_stage run completion (read-only).
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RUN="AP3_fusion_two_stage_3ds_s3407_20260501_110720"
METRICS="${PROJECT_DIR}/logs_accuracy_seq/${RUN}/metrics.csv"
CKPT="${PROJECT_DIR}/checkpoints_accuracy_seq/${RUN}/checkpoint_pretrain_epoch_49.pth"

echo "=== AP3 two_stage status ==="
python3 - <<'PY'
import csv, os, sys
metrics = os.environ["METRICS"]
with open(metrics) as f:
    val = [r for r in csv.DictReader(f) if r["phase"] == "val"]
eps = sorted({int(r["epoch"]) for r in val})
print(f"val epochs: {len(eps)} (min={min(eps)}, max={max(eps)})")
if max(eps) >= 49:
    print("STATUS: 50-epoch pretrain COMPLETE (epochs 0..49).")
else:
    print(f"STATUS: INCOMPLETE — need epochs through 49, currently max={max(eps)}")
    sys.exit(1)
PY
test -f "$CKPT" && echo "checkpoint_pretrain_epoch_49.pth: OK" || { echo "missing epoch_49 ckpt"; exit 1; }
