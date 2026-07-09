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
| **sdavt_meld_zh_agent** (B) | 中文 ASR 主导 | ❌ BLOCKED | checkpoint 目录不存在 |
| **meld_only** | 对照 | ❌ FAIL | AP1 checkpoint 与当前 `MultimodalEmotionModel` 架构不兼容（video_projection shape mismatch） |
| **ap2_m1** | 对照 | ❌ FAIL | 同上（旧 checkpoint 结构） |

**结论：** 当前仅 **R4 M3_M7** preset 可与最新模型代码正确加载；切换 preset 需先对齐 checkpoint 格式或重新导出权重。

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

1. **Preset B 阻塞**：`SDAVT_R4_M3_M7_chinese_agent` checkpoint 未训练/未落盘 → 需运行 `finetune_m3m7_chinese_agent.sh` 后再测。
2. **旧 preset 兼容**：`meld_only` / `ap2_m1` 与 R4 模型骨架不兼容 → 论文/demo 统一用 `sdavt_meld_v3_r4`。
3. **text-only degraded**：无 A/V 时 `degraded_mode=true`；在线采集应启用三模态以提升 raw 置信度。
4. **LLM**：Ollama 未启动时话术为 template 兜底；答辩前需 `start_ollama.sh`。
5. **smoke 测试**：`test_infer_and_agent_flow` 应改用合法 base64 或 text-only payload。

---

## 相关文档

- [`R4_CLOSEOUT_20260709.md`](R4_CLOSEOUT_20260709.md)
- [`EMOTION_AGENT_ENGINEERING_PLAN.md`](EMOTION_AGENT_ENGINEERING_PLAN.md) §14
- [`config_agent_deploy.yaml`](../config/config_agent_deploy.yaml)
