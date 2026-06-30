#!/usr/bin/env bash
# 归档 S1 日志，仅保留 S2（series=sdavt_v3_s2）run 供 TensorBoard 展示
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGDIR="$ROOT/logs_sdavt_v3"
ARCHIVE="$ROOT/logs_sdavt_v3_archived/s1_$(date +%Y%m%d)"

mkdir -p "$ARCHIVE"

shopt -s nullglob
for d in "$LOGDIR"/SDAVT_S1_* "$LOGDIR"/SDAVT_S1_*; do
  [[ -d "$d" ]] || continue
  echo "归档: $(basename "$d")"
  mv "$d" "$ARCHIVE/"
done

# 误重启的 S1 MOSEI（非 S2 前缀）
for d in "$LOGDIR"/SDAVT_S1_O0_mosei_*; do
  [[ -d "$d" ]] || continue
  echo "归档: $(basename "$d")"
  mv "$d" "$ARCHIVE/"
done

echo "保留 S2 run:"
ls -1 "$LOGDIR"/SDAVT_S2_* 2>/dev/null || echo "  (暂无)"
echo "归档目录: $ARCHIVE"
