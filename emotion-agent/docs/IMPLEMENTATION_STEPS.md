# Implementation Steps (Server Deploy + Local Capture)

## Step 1: Prepare Environment

1. Ensure Python 3.10+ and Node 20+ are available on server.
2. Clone repository and enter `emotion-agent/`.
3. Copy env:
   - `cd deploy`
   - `cp .env.example .env`

## Step 2: Start Backend

```bash
cd emotion-agent/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/api/v1/health
```

## Step 3: Start Frontend

```bash
cd emotion-agent/frontend
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```

Open browser at `http://<server-ip>:5173`.

## Step 4: Browser Capture and End-to-End Verification

1. Click `启用摄像头与麦克风`.
2. Input context text.
3. Click `执行一次推理与回复`.
4. Verify:
   - Emotion response appears in UI.
   - Agent reply appears in UI.
   - WebSocket event stream keeps updating.

## Step 5: Switch Providers

- Emotion provider:
  - `MODEL_PROVIDER=mock` (default)
  - `MODEL_PROVIDER=current` (use current training model adapter)
  - `MODEL_PROVIDER=external` (future external model)
- ASR provider:
  - `ASR_PROVIDER=mock`
  - `ASR_PROVIDER=whisper_api` (placeholder integration point)
- LLM provider:
  - `LLM_PROVIDER=template`
  - `LLM_PROVIDER=openai|ollama` (integration placeholder)

## Step 6: Integrate Current Trained Model

After training finishes in `project/`:

1. Set in env:
   - `MODEL_PROVIDER=current`
   - `MODEL_CONFIG_PATH=/workspace/project/config/...yaml`
   - `MODEL_CHECKPOINT_PATH=/workspace/project/checkpoints_accuracy_seq/...pth`
2. Restart backend service.
3. Re-run Step 4 verification.

## Step 7: Docker Deployment

```bash
cd emotion-agent/deploy
cp .env.example .env
docker compose up -d --build
```

Then open frontend from local browser and authorize local camera/mic.
