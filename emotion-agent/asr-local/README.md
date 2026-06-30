# asr-local (Whisper small, OpenAI-compatible)

本目录提供一个本地 ASR 服务，兼容 OpenAI 的转写接口：

- `POST /v1/audio/transcriptions`

用于配合 `emotion-agent/backend` 的 `ASR_PROVIDER=whisper_api` 直接对接。

## 依赖与前置

- Python 3.8+（建议独立 venv）
- **ffmpeg**（强烈推荐；浏览器录音常见为 `webm/opus`，无 ffmpeg 很难解码）

Ubuntu 安装：

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
```

## 启动（开发）

```bash
cd /mnt/sda1/lizhichun_24/code/multimodal/emotion-agent/asr-local
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./start_server.sh
# 或: uvicorn app:app --host 0.0.0.0 --port 9010
```

健康检查：

```bash
curl -sS http://127.0.0.1:9010/health
```

## 与 emotion-agent/backend 对接

编辑 `emotion-agent/backend/.env`：

```env
ASR_PROVIDER=whisper_api
ASR_WHISPER_API_URL=http://127.0.0.1:9010/v1/audio/transcriptions
ASR_WHISPER_API_KEY=
ASR_WHISPER_API_MODEL=small
ASR_WHISPER_API_LANGUAGE=zh
```

