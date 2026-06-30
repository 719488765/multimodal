# Figure 1 架构图元素详解（详细版 · 与代码一致）

> 文件：[`system_architecture_figure.svg`](system_architecture_figure.svg) · [`system_architecture_figure.html`](system_architecture_figure.html)

---

## (a) 离线工程 project/

### 数据采集与清洗

| 脚本 | 功能 |
|------|------|
| `download_crema_d.py` / `download_cmu_mosei_sdk.py` | 下载原始数据 |
| `organize_crema_d.py` / `organize_meld.py` / `organize_cmu_mosei_*.py` | 整理为统一目录 |
| `check_media_health.py` / `move_bad_media.py` | 质检与隔离坏样本 |
| `MultimodalDataset` | 加载 video/audio/text/labels，7 类映射 |

### 四种融合子模型内部结构

#### ① emotion_shift（CFN-ESA · AP2 主推荐）

```
V/A/T 加权求和
  → EmotionShiftAwareness
      · emotion_encoder (Linear)
      · BiLSTM shift_detector (2层双向)
      · shift_fusion
  → MultiheadAttention (Q=text, KV=video+audio+physio)
  → LayerNorm + 残差 + Dropout
```

文件：`emotion_shift.py` · val Best F1≈**0.562**

#### ② leader_follower（ICCV 2021）

```
YAML leader_modal (text|audio|video) 指定领导者
  → LeaderFollowerAttention ×3 (video/audio/physio 为跟随者)
      · Q←leader · K,V←follower
  → 4 模态 concat → final_fusion Linear → LayerNorm
```

文件：`leader_follower_attention.py` · AP3 消融 · leader_audio 适合大笑

#### ③ two_stage（GA2MIF）

```
4 模态为图节点
  → Stage1: GraphAttentionLayer × num_gat_layers (MultiheadAttention)
  → Stage2: CrossModalAttention + MultiHeadSelfAttention + FFN(GELU) × N
  → final_fusion Linear
```

文件：`two_stage_fusion.py` · AP3 对照 · 三混合易塌缩

#### ④ standard（基线）

```
4 模态 concat
  → MultiHeadSelfAttention × num_layers
  → CrossModalAttention × num_layers
  → FFN × num_layers
  → TemporalAttention 池化
```

文件：`attention_modules.py` · AP0 · F1≈0.52

### AP0–AP4 实验协议含义

| 阶段 | 全称含义 | 做什么 | 关键结果 |
|------|----------|--------|----------|
| **AP0** | Accuracy Plan 0 · 基线 | 三混合 + standard 融合 + 50 epoch | F1≈0.52，建立下界 |
| **AP1** | Accuracy Plan 1 · 单域上界 | crema/meld/mosei 各自单独训练 | MELD F1≈0.54 → **meld_only preset** |
| **AP2** | Accuracy Plan 2 · 主实验 | 三混合 + emotion_shift 配方消融 M1–M4 | **F1≈0.562** → **ap2_m1 preset** |
| **AP3** | Accuracy Plan 3 · 融合消融 | 仅换 fusion_strategy，其余 YAML 相同 | ES > LF > standard；two_stage 塌缩 |
| **AP4** | Accuracy Plan 4 · 域适应 | DANN + GradientReversalLayer，λ 扫描 | w=0.05 F1≈0.528，未超 AP2 |

产出：`checkpoints_accuracy_seq/*/checkpoint_pretrain_best_f1.pth`

---

## (b) 在线智能体 emotion-agent/

### 表现层

| 模块 | 技术 | 功能 |
|------|------|------|
| `App.jsx` | React 18 | 采集、preset、流水线 UI、概率条 |
| `api.js` | fetch + WS | 32KB 分块上传、health、隧道适配 |
| MediaRecorder | web API | video.webm ~3s |
| WebAudio | PCM→WAV | 16kHz mono |

### 编排层 FastAPI :8000

| 模块 | 功能 |
|------|------|
| `ModelRouter` | 加载 preset，禁止 silent mock |
| `CurrentProjectAdapter` | 桥接 project EmotionInferenceService |
| `ASRService` | HTTP → asr-local :9010 |
| `LLMService` | HTTP → Ollama :11434，注入 all_probs |
| `upload_buffer` | 分块重组 |
| `IngestBuffer` | 滑窗 3s step 1s |
| `emotion_arbitration.py` | final_emotion_label |
| `asr_emotion_calibration.py` | 第 5 步校准（project/utils） |

**六步流水线**：1接入 → 2ASR → 3合并 → 4推理 → 5校准 → 6仲裁 → LLM回复

### 微服务

| 服务 | 端口 | 技术 |
|------|------|------|
| ASR | 9010 | faster-whisper small · CUDA · zh |
| GPU 推理 | project | EmotionInferenceService · temporal_inference |
| LLM | 11434 | Ollama qwen2.5:7b-instruct |

**Presets**：meld_only★ · ap2_m1 · agent_chinese · mosei_only · ap4_w005

---

## (c) 推理示例

与 `routes.py _emotion_infer_core` 的 `pipeline_trace` 字段一一对应。

访问：`http://127.0.0.1:8765/system_architecture_figure.html`
