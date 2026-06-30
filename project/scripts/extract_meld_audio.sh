#!/usr/bin/env bash
# 从已整理的 MELD mp4 批量提取 mono 16kHz WAV（训练前必跑）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/home/lizhichun_24/.conda/envs/myenv310/bin/python}"
WORKERS="${WORKERS:-8}"

if ! "$PYTHON" -c "from data.meld_audio_utils import pyav_available, ffmpeg_available; import sys; sys.exit(0 if (pyav_available() or ffmpeg_available()) else 1)" 2>/dev/null; then
  echo "ERROR: 需要 PyAV (pip install av) 或 ffmpeg" >&2
  exit 1
fi

echo "==> MELD 音频提取 (PyAV/ffmpeg -> data/{train,val,test}/audio/meld_*.wav)"
exec env PYTHONUNBUFFERED=1 "$PYTHON" scripts/extract_meld_audio.py --workers "$WORKERS" "$@"
