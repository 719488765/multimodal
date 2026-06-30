# Emotion Agent

Emotion Agent is an online multimodal affective interaction system that is
decoupled from the training workflow under `project/`.

## Goals

- Capture camera and microphone input from a local browser client.
- Run server-side ASR, multimodal emotion inference, and LLM response generation.
- Keep model integration replaceable through adapter interfaces.
- Support both local demo and remote server deployment.

## Project Layout

- `backend/`: FastAPI services (ingest, ASR, emotion, LLM orchestration).
- `frontend/`: React browser client for realtime capture and visualization.
- `deploy/`: Docker and deployment configuration.
- `docs/`: runbooks and architecture notes.
- `desktop-client/`: phase-2 desktop client placeholder.

## 公网生产部署

- 文档：[deploy/SERVER_DEPLOY.md](deploy/SERVER_DEPLOY.md)（nginx + HTTPS + systemd）
- **详细操作（微调 → nginx → 测试）**：[docs/OPERATION_GUIDE_NGINX_FINETUNE_TEST.md](docs/OPERATION_GUIDE_NGINX_FINETUNE_TEST.md)
- 构建：`./scripts/build_production.sh`（`VITE_DEPLOY_MODE=server`，整包 multipart 上传 webm）
- 中文模型微调：`../project/scripts/finetune_agent_chinese.sh` → `apply_agent_chinese_preset.sh`

## Quick Start (Development)

**完整演示启动步骤见 [`START_DEMO.txt`](START_DEMO.txt)**（含 ASR、Ollama、单端口 8000、自检与排错）。

一键 tmux 启动：

```bash
cd emotion-agent
chmod +x scripts/start_all_demo.sh scripts/stop_all_demo.sh
./scripts/start_all_demo.sh
# 浏览器: http://127.0.0.1:8000
```

手动三分终端启动：

1. `./scripts/start_ollama.sh`
2. `cd asr-local && ./start_server.sh`
3. `./scripts/start_demo.sh` → 打开 `http://127.0.0.1:8000`

开发模式（可选，不推荐远程演示）：

1. Start backend:
   - `cd backend`
   - `python3 -m venv .venv && source .venv/bin/activate`
   - `pip install -r requirements.txt`
   - `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`
2. Start frontend:
   - `cd frontend`
   - `npm install`
   - `npm run dev -- --host 0.0.0.0 --port 5173`
3. Open `http://localhost:5173`.

## Runtime Switching

- Emotion provider: `MODEL_PROVIDER=mock|current|external`
- LLM provider: `LLM_PROVIDER=template|openai|ollama`
- ASR mode: `ASR_PROVIDER=mock|whisper_api`

All providers are abstracted behind interfaces so deployment can switch
implementations without breaking API contracts.
