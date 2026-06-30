# 情绪识别准确率审计与优化路线图

**版本**：2026-06-04（v2：补充本次落地清单与操作手册）  
**范围**：`project/` 训练与推理 + `emotion-agent/` 在线 Agent  
**关联文档**：
- [`emotion-agent/docs/ARCHITECTURE.md`](../../emotion-agent/docs/ARCHITECTURE.md)
- [`emotion-agent/docs/OPERATION_GUIDE_NGINX_FINETUNE_TEST.md`](../../emotion-agent/docs/OPERATION_GUIDE_NGINX_FINETUNE_TEST.md)
- [`MIXED_DATASET_TRAINING_ANALYSIS.md`](MIXED_DATASET_TRAINING_ANALYSIS.md)
- [`EXPERIMENT_ACCURACY_SEQ_MAIN_RECORD.md`](EXPERIMENT_ACCURACY_SEQ_MAIN_RECORD.md)

**维护约定**：每完成一阶段在文末「修订记录」追加一行，并更新 §5 Checklist 勾选状态。

---

## 1. Executive Summary

| 结论 | 说明 |
|------|------|
| 混合三数据集 val Best F1 **≈0.56** 已接近当前协议上限 | AP2_M1 emotion_shift（见实验记录） |
| `agent_chinese` 短微调 **未超越** ap2_m1 | 英文台词 + 中文 ASR 域偏移 |
| Agent 摄像头中文场景属于 **分布外（OOD）** | 扁平 softmax、Top 非开心属预期现象 |
| **单数据集部署**应优先 **MELD**（mp4→ResNet 与线上一致） | MOSEI 训练多用 `.npy` 特征，**不宜**直接作 webcam 权重 |
| 短期靠 **ASR 校准 + 多模态仲裁**；中期 **MELD 单域重训 + agent_capture** | 长期论文级融合/DA 消融 |

---

## 2. 问题树（为何 Agent 上不准）

### 2.1 数据与标签

- 三域异构（CREMA 实验室 / MELD 剧集 / MOSEI 网络视频）→ 混合 val 上限低。
- 7 类统一映射噪声：`surprise→anxious`、`disgust→other`；CREMA `neutral` 曾误映射至 id=5（**已在 `data/dataset.py` 修复**，旧 checkpoint 仍按旧分布学习）。
- 训练文本以 **英文** 为主；在线 ASR 为 **中文**。

### 2.2 模型与论文复现

| 融合策略 | 混合 val Best F1 | Agent 适用性 |
|----------|------------------|--------------|
| emotion_shift (CFN-ESA) | **0.562** (AP2_M1) | 在线多为 T=1，时序转移弱 |
| leader_text | 0.542 @ep6，Last 差 | 英文 text leader + 中文 ASR |
| leader_audio | 0.525 @ep43 较稳 | **单域重训推荐** |
| two_stage | **塌缩恒预测类 4** | **禁止部署** |
| standard | 0.518 | 可作对照 |

### 2.3 单域上界（选数据集依据）

| 单域 run | Best val Acc | Best val F1 | 与 Agent 视频管线 |
|--------|--------------|-------------|-------------------|
| MOSEI AVT+ES | **0.716** | **0.612** | 训练多为 `.npy` 特征，**≠** 在线 ResNet |
| **MELD AVT+ES** | **0.580** | **0.540** | **mp4 抽帧，与线上一致（推荐）** |
| CREMA AVT+ES | 0.345 | 0.304 | 实验室表演，与自拍差异大 |

### 2.4 在线 Agent 典型失败模式

- 模型输出 **扁平概率**（Top≈0.24），难以区分情绪。
- ASR「哈哈哈哈」与视觉笑容一致，但模型 Top 为 neutral/anxious/sad。
- 旧版 **5_asr_calibration: skipped**（happy 概率过低未触发规则）→ **已在本次优化中修复**。
- LLM 读 ASR 能判断开心，UI 仍显示模型 Top1 → **已加仲裁层**。

---

## 3. 本次 Plan 已完成的优化（代码/配置清单）

> 以下改动已在仓库中落地；**重启后端 + 浏览器强刷**后生效。

### 3.1 在线推理后处理（短期 P0）

| 改动 | 路径 | 作用 |
|------|------|------|
| ASR 校准增强 | [`project/utils/asr_emotion_calibration.py`](../utils/asr_emotion_calibration.py) | `flat_logits`（max&lt;0.38）、纯「哈哈」、Top 非 happy 时提升 happy；修正效价 |
| 多模态仲裁 | [`emotion-agent/backend/app/services/emotion_arbitration.py`](../../emotion-agent/backend/app/services/emotion_arbitration.py) | 合并模型 + 校准 + ASR 置信度 → `final_emotion_label` 供 UI/LLM |
| 流水线集成 | [`emotion-agent/backend/app/api/routes.py`](../../emotion-agent/backend/app/api/routes.py) | 步骤 **5_asr_calibration**、**6_arbitration**；trace 可追溯 |
| 响应字段 | [`emotion-agent/backend/app/models/schemas.py`](../../emotion-agent/backend/app/models/schemas.py) | `final_emotion_*`、`arbitration_*`、`is_flat_distribution` 等 |
| 部署配置 | [`project/config/config_agent_deploy.yaml`](../config/config_agent_deploy.yaml) | `default_preset: meld_only`；`asr_emotion_calibration` / `emotion_arbitration` 开关 |

### 3.2 权重 Preset 与 ModelRouter

| 改动 | 路径 | 作用 |
|------|------|------|
| 新增 preset | [`emotion-agent/backend/app/core/config.py`](../../emotion-agent/backend/app/core/config.py) | `meld_only`、`mosei_only`（原有 `ap2_m1`、`agent_chinese`、`ap4_w005`） |
| 懒加载多 preset | [`emotion-agent/backend/app/services/model_router.py`](../../emotion-agent/backend/app/services/model_router.py) | 前端 metadata 可切换权重；修复 `Path` 拼接 bug |
| 切换脚本 | [`project/scripts/apply_deploy_preset.sh`](../scripts/apply_deploy_preset.sh) | 一键写 `.env` 的 `MODEL_CHECKPOINT_PRESET` |
| 默认 preset | `emotion-agent/backend/.env` | **已执行** `apply_deploy_preset.sh meld_only` |

**Preset 与 checkpoint 对照：**

| preset | 训练 config | checkpoint（相对 project/） | val 参考 | 推荐场景 |
|--------|-------------|----------------------------|----------|----------|
| **meld_only** | `ap1_AVT_ES_meld_only_s3407.yaml` | `AP1_AVT_ES_pretrain_meld_only_.../checkpoint_pretrain_best_f1.pth` | F1≈0.54 | **Agent 默认** |
| ap2_m1 | `ap2_M1_effbatch8_ES_3ds_s3407.yaml` | `AP2_M1_ES_3ds_.../checkpoint_pretrain_best_f1.pth` | F1≈0.56 | 三混合对照 |
| mosei_only | `ap1_AVT_ES_mosei_only_s3407.yaml` | `AP1_AVT_ES_pretrain_mosei_only_.../checkpoint_pretrain_best_f1.pth` | Acc≈0.72 | 实验（npy 域差） |
| agent_chinese | `ap2_M1_chinese_text_agent.yaml` | `AP2_M1_chinese_text_agent/checkpoint_finetune_best_f1.pth` | F1≈0.54 | 中文 BERT 试验 |

### 3.3 前端采集与上传

| 改动 | 路径 | 作用 |
|------|------|------|
| 多帧采样 | [`emotion-agent/frontend/src/App.jsx`](../../emotion-agent/frontend/src/App.jsx) | 录制每 0.75s 一帧（最多 4 张），`capture_frames_b64` 上传 |
| preset 下拉 | 同上 | 默认 `meld_only`；请求带 `metadata.checkpoint_preset` |
| 仲裁/校准展示 | 同上 | 显示 `arbitration_source`、扁平分布提示 |
| Cloudflare 上传修复 | [`emotion-agent/frontend/src/api.js`](../../emotion-agent/frontend/src/api.js) | 同源 API、禁止错误 `:8000`、隧道禁止分块 |
| 生产构建 | `emotion-agent/frontend/dist/` | 当前 bundle：`index-BHqvHs58.js`（**强刷后确认 Network 中文件名**） |

### 3.4 后端时序推理

| 改动 | 路径 | 作用 |
|------|------|------|
| 按窗多帧视频 | [`project/utils/emotion_inference_service.py`](../utils/emotion_inference_service.py) | `multi_frame_sequence`；短 JPEG 采集偏好单窗 |
| 时序参数 | [`project/config/config_agent_deploy.yaml`](../config/config_agent_deploy.yaml) | `stride_sec: 1.5`；`jpeg_prefer_single_window: true` |

### 3.5 训练侧准备（中期，脚本已就绪）

| 改动 | 路径 | 作用 |
|------|------|------|
| CREMA neutral→4 | [`project/data/dataset.py`](../data/dataset.py) | 标签映射与标准 7 类一致 |
| MELD Agent 训练 yaml | [`project/config/rerun/accuracy_plan/ap2_M1_meld_only_agent.yaml`](../config/rerun/accuracy_plan/ap2_M1_meld_only_agent.yaml) | 单域 MELD + effbatch8 + **leader_audio** |
| 中文 v2 微调 yaml | [`ap2_M1_chinese_text_agent_v2.yaml`](../config/rerun/accuracy_plan/ap2_M1_chinese_text_agent_v2.yaml) | 早停、CREMA 映射修复（可选） |
| 训练脚本 | [`train_meld_agent.sh`](../scripts/train_meld_agent.sh)、[`finetune_agent_chinese_v2.sh`](../scripts/finetune_agent_chinese_v2.sh) | GPU 训练入口 |
| 标签审计 | [`audit_label_distribution.py`](../scripts/audit_label_distribution.py) | 统计 train/val 类别分布 |
| Agent 采集 | [`data/agent_capture/README.md`](../data/agent_capture/README.md)、[`organize_agent_capture.py`](../scripts/organize_agent_capture.py) | 闭环数据目录与整理 |
| 回归脚本 | [`eval_agent_capture_cases.py`](../scripts/eval_agent_capture_cases.py) | 开心/哈哈/中性用例 |
| 单元测试 | [`tests/test_asr_emotion_calibration.py`](../tests/test_asr_emotion_calibration.py)、[`emotion-agent/backend/tests/test_emotion_arbitration.py`](../../emotion-agent/backend/tests/test_emotion_arbitration.py) | 4+1 项通过 |

### 3.6 文档更新

- 本文档（v2）
- [`emotion-agent/docs/ARCHITECTURE.md`](../../emotion-agent/docs/ARCHITECTURE.md) §0 部署与后处理
- [`emotion-agent/docs/OPERATION_GUIDE_NGINX_FINETUNE_TEST.md`](../../emotion-agent/docs/OPERATION_GUIDE_NGINX_FINETUNE_TEST.md) 步骤 G

---

## 4. 【现在】你需要立即执行的步骤

> 按顺序执行；每步后打勾。预计 10–15 分钟（不含 GPU 训练）。

### 步骤 A：释放端口并重启服务

```bash
# 若 8000 被占用
fuser -k 8000/tcp 2>/dev/null || true

cd /home/lizhichun_24/sda1/code/multimodal/emotion-agent
FORCE_RESTART=1 ./scripts/start_demo.sh
```

**期望日志：**

- `Application startup complete`（无 `address already in use`）
- `preset=meld_only` 或 `Loaded emotion model preset=meld_only`
- ASR / LLM startup ok

### 步骤 B：健康检查

```bash
curl -s http://127.0.0.1:8000/api/v1/health | python3 -m json.tool
```

**期望：**

- `"ok": true`
- `"preset": "meld_only"`（或 model 段含 meld_only）
- `"using_trained_checkpoint": true`

若 preset 不是 meld_only：

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project
./scripts/apply_deploy_preset.sh meld_only
# 再重复步骤 A
```

### 步骤 C：浏览器验证

1. 打开 `http://127.0.0.1:8000`（Cloudflare 隧道则用隧道地址，**不要加 :8000**）。
2. **Ctrl+Shift+R** 强刷；DevTools → Network 确认 JS 为 **`index-BHqvHs58.js`**（勿用旧版 `index-DsJ-lzQl.js`）。
3. 会话状态 → 推理权重应为 **meld_only**。
4. 录制 5–10 秒，说「哈哈哈哈」或「我很高兴」并微笑 → **结束并推理**。

**期望结果：**

| 检查项 | 期望 |
|--------|------|
| 情感结果 | **开心**（或 happy 概率最高） |
| 流水线 5_asr_calibration | `applied` 或 profile 含 `flat_logits` |
| 流水线 6_arbitration | `applied` 或 `passthrough`；final=happy |
| 识别文本 | ASR 含「哈哈」或开心语义 |

### 步骤 D：本地回归（可选，1 分钟）

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project
python3 scripts/eval_agent_capture_cases.py
python3 -m pytest tests/test_asr_emotion_calibration.py -q

cd ../emotion-agent/backend
python3 -m pytest tests/test_emotion_arbitration.py -q
```

**期望：** 全部 OK / passed。

### 步骤 E：A/B 对照（可选）

页面切换推理权重为 **ap2_m1** 或 **agent_chinese**，同场景再测一次，记录 Top1 与是否需仲裁才得到 happy（便于论文/答辩对比表）。

---

## 5. 【后续】分阶段操作步骤

### 阶段 1：MELD 单域重训（中期，需 GPU，约 1–2 天）

**目的**：在 MELD 单域上复现/超越 AP1（F1≈0.54），并使用 **leader_audio** 更贴合大笑/声学。

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project
export PYTHON=/home/lizhichun_24/.conda/envs/myenv310/bin/python

# 建议 tmux 后台
tmux new -s meld_agent
HF_HUB_OFFLINE=0 ./scripts/train_meld_agent.sh
# Ctrl+B D 脱离；tail -f logs_accuracy_seq/AP2_M1_meld_only_agent_*/train.log
```

**训练完成后：**

```bash
# 1. 找到 best ckpt（示例路径，以实际 run 目录为准）
ls -lt checkpoints_accuracy_seq/AP2_M1_meld_only_agent_*/checkpoint_pretrain_best_f1.pth | head -1

# 2. 链到固定 preset 目录（便于 deploy 加载）
mkdir -p checkpoints_accuracy_seq/AP2_M1_meld_only_agent
ln -sf "$(readlink -f checkpoints_accuracy_seq/AP2_M1_meld_only_agent_YYYYMMDD_HHMMSS/checkpoint_pretrain_best_f1.pth)" \
  checkpoints_accuracy_seq/AP2_M1_meld_only_agent/checkpoint_pretrain_best_f1.pth

# 3. 在 config.py / config_agent_deploy.yaml 增加 preset meld_agent 指向新 ckpt（或覆盖 meld_only 链）
# 4. 切换并重启
./scripts/apply_deploy_preset.sh meld_only
cd ../emotion-agent && FORCE_RESTART=1 ./scripts/start_demo.sh
```

**验收：** 混合 val Best F1 ≥ 0.55；Agent 场景「哈哈/开心」命中率高于旧 meld_only ckpt。

---

### 阶段 2：Agent 采集闭环微调（中期，需人工录制 + GPU）

**目的**：用真实浏览器采集数据纠正 webcam + 中文 ASR 域偏移。

1. **阅读** [`data/agent_capture/README.md`](../data/agent_capture/README.md)。
2. **录制** ≥50 条（每类 happy/sad/neutral 至少 15 条）：
   - wav 16kHz mono + 短 webm 或 4 帧截图 + 标签 txt（第一行：`happy`/`sad`/`neutral` 等）。
3. **整理：**

```bash
python3 scripts/organize_agent_capture.py --src /path/to/raw_recordings --split train
python3 scripts/organize_agent_capture.py --src /path/to/raw_val --split val
```

4. **修改 yaml**（复制 `ap2_M1_meld_only_agent.yaml`）：

```yaml
finetune:
  datasets: ["meld", "agent_capture"]  # 需在 dataset 加载逻辑中支持 agent_capture 目录
  epochs: 5
```

5. **短微调**（从阶段 1 best ckpt resume）并重复「链 ckpt → apply preset → 重启 → 浏览器验收」。

---

### 阶段 3：标签与数据审计（随时可做）

```bash
cd project
python3 scripts/audit_label_distribution.py \
  --config config/rerun/accuracy_plan/ap2_M1_meld_only_agent.yaml
```

关注 neutral 是否过多、happy 是否过少；对照 ClassBalanced 是否导致扁平 softmax。

---

### 阶段 4：中文 BERT 路线（可选，长期）

若坚持中文 ASR 进 BERT：

```bash
HF_HUB_OFFLINE=0 ./scripts/finetune_agent_chinese_v2.sh
./scripts/apply_deploy_preset.sh agent_chinese
```

**注意：** 三混合 + 英文台词微调对纯中文 ASR 提升有限；更推荐 **MELD 单域 + agent_capture** 或 **中文文本机翻子集** 后再训。

---

### 阶段 5：长期研究与论文（P2）

| 任务 | 操作 |
|------|------|
| MELD 单域 + DA w=0.05 | 复制 AP4 yaml，改 `datasets: ["meld"]` 重跑 |
| 禁用 two_stage | 勿用于部署；若修复 GAT 塌缩需重跑 AP3 |
| 轻量中文情感头 | bert-base-chinese 3 类 + 与主模型 logit 加权（λ≈0.3） |
| TensorBoard 论文图 | 导出 AP0–AP4 的 val/accuracy、f1、cls_ce_unweighted 四宫格 |
| 答辩对照表 | 同一句「哈哈」在 meld_only / ap2_m1 / agent_chinese 下 Top3 概率 |

---

## 6. 优化路线图总览

| 阶段 | 状态 | 内容 |
|------|------|------|
| **P0 短期** | **已完成** | ASR 校准、仲裁、preset、多帧、上传修复、前端 build |
| **P1 中期** | **待你执行** | MELD 重训、agent_capture 采集与微调 |
| **P2 长期** | 规划中 | DA 单域、中文情感头、论文图表 |

---

## 7. 验收标准

| 场景 | 通过条件 |
|------|----------|
| ASR「哈哈哈哈」+ 扁平概率 | `final_label=happy`；步骤 5 或 6 非 skipped |
| ASR「我很难过」 | sad 提升；非长期 neutral |
| `eval_agent_capture_cases.py` | 全绿 |
| MELD 单域重训后 | val Best F1 ≥ 0.55；Agent 开心场景明显改善 |

---

## 8. 实施 Checklist（总进度）

### 已完成（开发侧）

- [x] 本文档 v2
- [x] `asr_emotion_calibration` 增强（flat_logits / 纯哈哈 / 非 neutral Top）
- [x] `emotion_arbitration.py` + routes + schema + 前端展示
- [x] preset：`meld_only`（默认）/ `mosei_only` / `ap2_m1` + `apply_deploy_preset.sh`
- [x] `ModelRouter` 多 preset 懒加载 + Path 修复
- [x] 多帧采集 + 时序单窗策略 + Cloudflare 上传修复
- [x] CREMA neutral→4（`dataset.py`）
- [x] `ap2_M1_meld_only_agent.yaml`、`train_meld_agent.sh`
- [x] `audit_label_distribution.py`、`organize_agent_capture.py`、`agent_capture/README.md`
- [x] 回归测试与 eval 脚本全绿
- [x] 前端 build `index-BHqvHs58.js`
- [x] `.env` → `MODEL_CHECKPOINT_PRESET=meld_only`（已执行 apply 脚本）

### 待你执行（操作侧）

- [ ] **A** 重启 backend（§4 步骤 A–C）
- [ ] **B** 浏览器强刷并验证「哈哈→开心」+ 流水线 5/6
- [ ] **C** 运行回归脚本（§4 步骤 D）
- [ ] **D** GPU：`./scripts/train_meld_agent.sh`（§5 阶段 1）
- [ ] **E** 采集 agent_capture ≥50 条并微调（§5 阶段 2）
- [ ] **F** 可选：ap2_m1 / agent_chinese A/B 对照记录（§4 步骤 E）

---

## 9. 常见问题

| 现象 | 处理 |
|------|------|
| `stripBase64Payload is not defined` | 强刷；确认 JS 为 `index-BHqvHs58.js` |
| `address already in use :8000` | `fuser -k 8000/tcp` 后重启 |
| 5_asr_calibration skipped | 确认 ASR 有文本；升级后纯「哈哈」应触发 flat_logits |
| 仍显示平静 | 看步骤 6 是否 arbitration→happy；试 preset ap2_m1 对照 |
| Failed to fetch（Cloudflare） | 勿用 `:8000`；录制 ≤10s；强刷 |
| mosei_only 在线仍差 | 预期内（npy 训练 vs ResNet 在线） |

---

## 10. 修订记录

| 日期 | 摘要 |
|------|------|
| 2026-06-04 | v1：审计结论、单域 MELD 策略、P0–P2 路线图 |
| 2026-06-04 | **v2**：补充本次 Plan 全部落地清单（§3）、立即操作手册（§4）、后续分阶段步骤（§5）、Checklist 与 FAQ |
