# Emotion Agent 测试报告（2026-07-09）

> R4 close-out 后 Phase 0–3 系统化测试记录。默认 preset：**sdavt_meld_v3_r4**（M3_M7，F1=0.696）。

---

## Phase 0 — 基线确认

| 检查项 | 结果 |
|--------|------|
| `verify_trained_model.sh` | ✅ `using_trained_checkpoint=true` |
| `/api/v1/health` | ✅ `ok=true`，模型 loaded |
| `/api/v1/model/status` | ✅ preset=`sdavt_meld_v3_r4`，checkpoint 存在 |
| 冷启动（模型加载） | ~16s（ResNet50 + wav2vec-large + roberta-base → CUDA） |
| 热推理 latency（text-only） | min **25ms**，avg **32ms**，max **37ms**（6 次采样） |

**文本 prob 分布（preset A，text-only，degraded_mode=true 因无 A/V）：**

| 输入 | 模型 raw | ASR 校准后 | sad prob |
|------|----------|------------|----------|
| `I am so sad today` | angry | **sad** | **0.486** |
| `我很难过` | happy | **sad** | **0.508** |

> 文本单模态 raw 分布偏 flat；ASR 校准链将 sad 类提升到 >0.25，符合 OPERATION_GUIDE 验收口径。

---

## Phase 1 — 离线回归

| 脚本 | 结果 |
|------|------|
| `tests/test_emotion_arbitration.py` | ✅ **3/3** pass |
| `app/tests/test_smoke.py` | ⚠️ **1/2** pass（`test_infer_and_agent_flow` 因测试 payload 使用非法 base64 `"audio"` 触发 ASR decode 失败；health 测试通过） |
| `eval_zh_agent_benchmark.py` | ✅ **30/30**（calibrated=30/30，final=30/30） |
| `eval_agent_capture_cases.py` | ✅ **30/30** |

---

## Phase 2 — Preset A/B

| Preset | 场景 | 结果 | 备注 |
|--------|------|------|------|
| **sdavt_meld_v3_r4** (A) | 英文/中文 text | ✅ PASS | sad prob 0.486 / 0.508 |
| **sdavt_meld_zh_agent** (B) | 中文 ASR 主导 | ✅ PASS（**finetune ckpt**） | sad=**0.488** / 0.473（2026-07-13 重测，val F1=0.601） |
| **meld_only** | 对照 | ✅ PASS | legacy remap 后 sad=0.544 |
| **ap2_m1** | 对照 | ✅ PASS | legacy remap 后 sad=0.535 |

**结论：** `remap_legacy_checkpoint_state_dict` 已修复旧 AP1/AP2 checkpoint 加载；Preset B 已完成 M3_M7 中文 finetune 并部署。

### Preset B 重测（2026-07-13，finetune 完成后）

| 检查项 | bootstrap（07-09） | finetune 后（07-13） |
|--------|-------------------|----------------------|
| 离线 val F1 | — | **0.601**（epoch 9，early-stop @13） |
| `eval_zh_agent_benchmark.py` | 30/30 | **30/30** |
| `eval_agent_capture_cases.py` | 30/30 | **30/30** |
| infer-upload `我很难过` sad prob | 0.164 | **0.488**（final=sad） |
| infer-upload `I am so sad today` sad prob | — | **0.473**（final=sad） |
| 默认 preset / checkpoint | bootstrap | `sdavt_meld_zh_agent` → `checkpoint_finetune_best_f1.pth` |

### Preset B v2（2026-07-13，中文策略 + 二阶段微调）

| 检查项 | v1 | v2 |
|--------|----|----|
| 离线 val F1 | 0.601 | **0.6054**（epoch 10，256-sample finetune + zh 增强数据） |
| 训练数据 | MELD 英文字幕 | MELD + 500 `*_zh.txt` + 97 agent_capture 注入 |
| `eval_zh_agent_benchmark.py` | 30/30 | **30/30** |
| `eval_agent_capture_cases.py` | 30/30 | **30/30** |
| 自动路由中文 preset | `sdavt_meld_zh_agent` | **`sdavt_meld_zh_agent_v2`** |
| 默认部署 preset | v1 | **v2**（F1 优于 v1） |

**工程增强（同批）：** 前端「按语言自动选模型」+ optgroup 分组；`metadata.auto_preset`；`PRESET_METADATA.language/group`。

---

## Phase 3 — E2E 在线测试

**启动方式：** 后端 `./start_server.sh`（:8000）+ ASR `./start_server.sh`（:9010）；未启 Ollama（LLM 降级 template）。

| 检查项 | 结果 |
|--------|------|
| `/api/v1/health` asr_ok | ✅（ASR 启动后） |
| `infer-upload` text=`我很难过` | ✅ label=sad，sad=0.508 |
| `infer-upload` text=`I am so sad today` | ✅ label=sad，sad=0.486 |
| `pipeline_trace` 四步链 | ✅ modalities → model → asr_calibration → arbitration |
| `final_emotion_label` vs `model_emotion_label` | ✅ sad（final）← happy（model）经校准 |

**pipeline_trace 示例（text-only）：**
- model raw: happy @ conf≈0.19
- asr_calibration: happy → sad（negative_asr_model_conflict）
- arbitration: final=sad

---

## 已知问题与建议

1. **中文 finetune**：✅ 已完成（2026-07-10 early-stop，best val F1=0.601）；Preset B 已部署并重测 PASS。
2. **旧 preset**：`meld_only` / `ap2_m1` 已通过 `utils/helpers.py::remap_legacy_checkpoint_state_dict` 自动 remap。
3. **text-only degraded**：无 A/V 时 `degraded_mode=true`；在线采集应启用三模态以提升 raw 置信度。
4. **LLM**：Ollama 未启动时话术为 template 兜底；答辩前需 `start_ollama.sh`。
5. **smoke 测试**：`test_infer_and_agent_flow` 应改用合法 base64 或 text-only payload。

---

## 相关文档

- [`R4_CLOSEOUT_20260709.md`](R4_CLOSEOUT_20260709.md)
- [`EMOTION_AGENT_ENGINEERING_PLAN.md`](EMOTION_AGENT_ENGINEERING_PLAN.md) §14
- [`config_agent_deploy.yaml`](../config/config_agent_deploy.yaml)
