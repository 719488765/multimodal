# Emotion Agent Runbook

## 1. Development Mode

### Backend

```bash
cd emotion-agent/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd emotion-agent/frontend
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```

Open `http://localhost:5173`.

## 2. Remote Server Deployment (Browser First)

1. Copy env template:
   - `cd emotion-agent/deploy`
   - `cp .env.example .env`
2. Edit `.env`:
   - `MODEL_PROVIDER=mock` for integration test.
   - Later set `MODEL_PROVIDER=current` and configure checkpoint path.
3. Start containers:
   - `docker compose up -d --build`
4. Expose ports:
   - Backend: `${BACKEND_PORT}` (default `8000`)
   - Frontend: `${FRONTEND_PORT}` (default `5173`)

Local machine uses browser access to frontend URL and grants camera/microphone.

## 3. Integration with Current Training Model

When training artifacts are ready:

- Set `MODEL_PROVIDER=current`.
- Set `MODEL_CONFIG_PATH` to config under `project/config/...`.
- Set `MODEL_CHECKPOINT_PATH` to exported checkpoint path.

No API changes are required; only adapter environment changes.

## 4. Fallback/Recovery

- If ASR is unstable, keep `ASR_PROVIDER=mock` for demo continuity.
- If LLM is unstable, keep `LLM_PROVIDER=template`.
- If current model is not ready, keep `MODEL_PROVIDER=mock` or `external`.
