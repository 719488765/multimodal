#!/usr/bin/env bash
# Local Whisper ASR (faster-whisper), OpenAI-compatible /v1/audio/transcriptions
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PORT="${PORT:-9010}"
HOST="${HOST:-0.0.0.0}"
WHISPER_MODEL="${WHISPER_MODEL:-small}"
WHISPER_BEAM_SIZE="${WHISPER_BEAM_SIZE:-5}"
WHISPER_FAST_BEAM_SIZE="${WHISPER_FAST_BEAM_SIZE:-3}"

# 自动选择设备：双卡时 ASR 默认 GPU1（GPU0 常被微调/情绪模型占用）
if [[ "${WHISPER_DEVICE:-auto}" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    GPU_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
    if [[ "${GPU_COUNT:-0}" -ge 2 ]]; then
      WHISPER_DEVICE=cuda
      WHISPER_DEVICE_INDEX="${WHISPER_DEVICE_INDEX:-1}"
      WHISPER_COMPUTE_TYPE="${WHISPER_COMPUTE_TYPE:-float16}"
    else
      FREE_MB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 2>/dev/null | head -1 | tr -d ' ')"
      if [[ "${FREE_MB:-0}" -gt 4096 ]]; then
        WHISPER_DEVICE=cuda
        WHISPER_DEVICE_INDEX="${WHISPER_DEVICE_INDEX:-0}"
        WHISPER_COMPUTE_TYPE="${WHISPER_COMPUTE_TYPE:-float16}"
      else
        WHISPER_DEVICE=cpu
        WHISPER_DEVICE_INDEX="${WHISPER_DEVICE_INDEX:-0}"
        WHISPER_COMPUTE_TYPE="${WHISPER_COMPUTE_TYPE:-int8}"
        WHISPER_MODEL="${WHISPER_MODEL:-tiny}"
      fi
    fi
  else
    WHISPER_DEVICE=cpu
    WHISPER_DEVICE_INDEX="${WHISPER_DEVICE_INDEX:-0}"
    WHISPER_COMPUTE_TYPE="${WHISPER_COMPUTE_TYPE:-int8}"
  fi
else
  WHISPER_DEVICE="${WHISPER_DEVICE}"
  WHISPER_DEVICE_INDEX="${WHISPER_DEVICE_INDEX:-0}"
  WHISPER_COMPUTE_TYPE="${WHISPER_COMPUTE_TYPE:-int8}"
  if [[ "$WHISPER_DEVICE" == "cuda" ]]; then
    WHISPER_COMPUTE_TYPE="${WHISPER_COMPUTE_TYPE:-float16}"
  fi
fi

if [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
  PYTHON="$SCRIPT_DIR/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

# 优先使用 myenv310：与 PyTorch 共用 cuDNN，避免 .venv 在 GPU 上 libcudnn 崩溃
CONDA_PY="${CONDA_PY:-/home/lizhichun_24/.conda/envs/myenv310/bin/python}"
if [[ -x "$CONDA_PY" ]] && "$CONDA_PY" -c "import faster_whisper" 2>/dev/null; then
  PYTHON="$CONDA_PY"
elif [[ -x "$CONDA_PY" ]]; then
  echo "Installing faster-whisper into $CONDA_PY ..."
  "$CONDA_PY" -m pip install -q -r requirements.txt
  PYTHON="$CONDA_PY"
fi

if ! "$PYTHON" -c "import faster_whisper" 2>/dev/null; then
  echo "Installing asr-local dependencies into $PYTHON ..."
  "$PYTHON" -m pip install -r requirements.txt
fi

export WHISPER_MODEL WHISPER_DEVICE WHISPER_DEVICE_INDEX WHISPER_COMPUTE_TYPE WHISPER_BEAM_SIZE WHISPER_FAST_BEAM_SIZE
echo "Starting ASR http://${HOST}:${PORT} model=${WHISPER_MODEL} device=${WHISPER_DEVICE} index=${WHISPER_DEVICE_INDEX}"
exec "$PYTHON" -m uvicorn app:app --host "$HOST" --port "$PORT"
