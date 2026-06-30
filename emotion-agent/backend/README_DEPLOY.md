# Emotion Agent Backend — 真实模型部署

## 前置条件

- GPU（推荐 RTX 4090 或同级，显存 ≥ 8GB）
- 已训练 checkpoint 位于 `project/checkpoints_accuracy_seq/`
- HuggingFace 缓存可用（建议 `HF_HUB_OFFLINE=1` 使用本地缓存）
- Python 环境与训练一致（如 `myenv310`）

## 安装依赖

使用与训练相同的 conda 环境（如 `myenv310`），**不要**依赖系统自带的 `uvicorn` 命令：

```bash
conda activate myenv310
cd emotion-agent/backend
pip install -r requirements.txt
```

或指定 Python 路径：

```bash
/home/lizhichun_24/.conda/envs/myenv310/bin/pip install -r requirements.txt
```

## 环境变量（`.env`）

| 变量 | 说明 |
|------|------|
| `MODEL_PROVIDER` | `current` 启用真实推理；`mock` 为关键词占位 |
| `MODEL_CHECKPOINT_PRESET` | `ap2_m1`（默认，Best F1≈0.562）或 `ap4_w005`（DA 线，F1≈0.528） |
| `PROJECT_ROOT` | 指向 `multimodal/project` 绝对路径 |
| `MODEL_DEVICE` | `cuda` / `cpu` |
| `MODEL_FAIL_ON_ERROR` | `true` 时推理失败返回 503（答辩演示推荐） |

路径可由 preset 自动解析；若需手动指定：

```bash
MODEL_CONFIG_PATH=/path/to/project/config/rerun/accuracy_plan/ap2_M1_effbatch8_ES_3ds_s3407.yaml
MODEL_CHECKPOINT_PATH=/path/to/project/checkpoints_accuracy_seq/.../checkpoint_pretrain_best_f1.pth
```

## 启动

**推荐**（一键脚本，自动用 myenv310 + `python -m uvicorn`）：

```bash
cd emotion-agent/backend
./start_server.sh
```

或手动：

```bash
conda activate myenv310
cd emotion-agent/backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

若直接运行 `uvicorn ...` 出现 `Command 'uvicorn' not found`，说明当前 shell 未激活 conda 环境，请改用上面的 `python -m uvicorn`。

启动时会加载 checkpoint（约 1–2 分钟）。检查：

```bash
curl http://localhost:8000/api/v1/health
```

应看到 `"model_provider": "current"` 且 `"loaded": true`，以及 **`"asr_ok": true`**。

## ASR（语音转写，非 mock）

此前 `ASR_PROVIDER=mock` 会固定返回「我感觉还好，稍微有点紧张。」——**不是真实识别**。答辩演示请使用 Whisper：

**终端 1 — ASR 服务（端口 9010）**

```bash
cd emotion-agent/asr-local
chmod +x start_server.sh
./start_server.sh
```

**终端 2 — 情绪后端（端口 8000）**

`.env` 中应包含：

```env
ASR_PROVIDER=whisper_api
ASR_WHISPER_API_URL=http://127.0.0.1:9010/v1/audio/transcriptions
```

检查：

```bash
curl -s http://127.0.0.1:9010/health
curl -s http://127.0.0.1:8000/api/v1/health | python3 -m json.tool
# asr_ok 应为 true；推理响应 asr_provider 应为 whisper_api
```

## LLM（生成式话术，非 template）

`.env` 默认 `LLM_PROVIDER=ollama`。需 **Ollama 已启动且已拉取模型**（首次约 4.7GB）：

```bash
cd emotion-agent
chmod +x scripts/start_ollama.sh
./scripts/start_ollama.sh
```

检查：

```bash
curl -s http://127.0.0.1:11434/api/tags
curl -s http://127.0.0.1:8000/api/v1/health | python3 -c "import sys,json; d=json.load(sys.stdin); print('llm_ok',d.get('llm_ok'))"
```

- `llm_ok: true` → 助手回复来源应为 **`ollama`**
- `llm_ok: false` 或 回复来源 **`template`** → Ollama 未就绪，仅为模板兜底（非生成式）

## 如何确认使用的是训练模型（非 mock）

### 一键检查

```bash
chmod +x scripts/verify_trained_model.sh
./scripts/verify_trained_model.sh http://127.0.0.1:8000
```

或：

```bash
curl -s http://127.0.0.1:8000/api/v1/model/status | python3 -m json.tool
```

**训练模型已就绪** 当且仅当：

| 检查项 | 训练模型 | mock |
|--------|----------|------|
| `using_trained_checkpoint` | `true` | `false` |
| `model.provider` | `current` | `mock` |
| `model.loaded` | `true` | 任意 |
| 启动日志 | `[EMOTION_MODEL] startup ok provider=current` | `NOT using trained checkpoint` |

### 单次推理响应字段（`/api/v1/emotion/infer`）

| 字段 | 训练模型典型值 | mock 典型值 |
|------|----------------|-------------|
| `model_provider` | `current` | `mock` |
| `inference_source` | **`checkpoint`** | `mock_heuristic` 或 `mock_fallback` |
| `checkpoint_preset` | `ap2_m1` / `ap4_w005` | 空 |
| `fusion_strategy` | `emotion_shift` 或 `standard` | `none` |
| `checkpoint_file` | `checkpoint_pretrain_best_f1.pth` | 空 |
| `inference_ms` | 通常 **>50ms**（GPU） | `0` |

前端情绪卡片底部会显示：`推理：checkpoint · 预设 ap2_m1 · emotion_shift · 120ms · checkpoint_pretrain_best_f1.pth`。

### 后端日志关键字

启动成功：

```text
[EMOTION_MODEL] startup ok provider=current preset=ap2_m1 ...
[TRAINED_MODEL] loaded checkpoint=checkpoint_pretrain_best_f1.pth fusion=emotion_shift ...
```

每次推理成功：

```text
[TRAINED_MODEL] forward label=neutral id=4 conf=0.512 ms=85.3 ...
[EMOTION_MODEL] infer ok source=checkpoint provider=current ...
```

若出现以下任一条，说明**不是**训练模型：

```text
[EMOTION_MODEL] startup provider=mock ...
[EMOTION_MODEL] infer degraded to mock_fallback ...
[emotion_infer] NOT using trained checkpoint ...
```

### mock 与训练模型的行为差异（辅助判断）

- **mock**：`inference_ms=0`；无 `checkpoint_file`；说「开心」几乎必为 `happy`（关键词规则）；`valence/arousal` 每次随机跳动明显。
- **训练模型**：`inference_source=checkpoint`；有 `emotion_id` 与 7 维 `all_probs`（softmax，和≈1）；换表情/语气时标签会连续变化而非纯关键词。

### `.env` 必查项

```bash
grep -E '^MODEL_PROVIDER|^MODEL_CHECKPOINT_PRESET' .env
# 应为：
# MODEL_PROVIDER=current
# MODEL_CHECKPOINT_PRESET=ap2_m1
```

修改 `.env` 后必须**重启**后端进程。

## 切换 checkpoint

修改 `.env` 中 `MODEL_CHECKPOINT_PRESET=ap4_w005` 后**重启**服务。

## CLI 对照（同一推理运行时）

```bash
cd project
python scripts/inference.py \
  --config config/rerun/accuracy_plan/ap2_M1_effbatch8_ES_3ds_s3407.yaml \
  --model_path checkpoints_accuracy_seq/AP2_M1_ES_3ds_effbatch8_s3407_20260422_210615/checkpoint_pretrain_best_f1.pth \
  --video /path/to/video.mp4 \
  --audio /path/to/audio.wav \
  --text "optional"
```

## 测试

```bash
# 默认 mock，不加载 GPU
pytest app/tests/

# 可选 GPU 加载测试
RUN_GPU_TESTS=1 pytest app/tests/test_inference_utils.py -m gpu
```
