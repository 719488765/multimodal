# SDAVT v3 R4 完整实验报告

*初版生成：2026-07-09（`scripts/build_r4_full_experiment_report.py`）*  
*v1.1（2026-07-16）：纳入中文微调 v1/v2、消融解读、Agent Preset 矩阵*  
*v1.2（2026-07-16）：**对照 `logs_sdavt_v3_r4/` 全量 59 组 `metrics.csv` 复核** Best/Last、中文逐 epoch、稳定性分析；本文件为 R4+微调**权威长文**，并作为 emotion-agent 前端模型列表参考*

**数据源**：`outputs_sdavt_v3_r4/experiment_queue.json`（55 jobs）+ Close-out + `logs_sdavt_v3_r4/*/metrics.csv`（含中文 v1/v2）+ `outputs_sdavt_v3_r4/status/m3m7_zh_*.log`

**配套文档**：自动简表 [`SDAVT_V3_R4_EXPERIMENT_RESULTS.md`](SDAVT_V3_R4_EXPERIMENT_RESULTS.md)；论文总控 [`THESIS_EXPERIMENT_MASTER_SUMMARY.md`](THESIS_EXPERIMENT_MASTER_SUMMARY.md)。

---

## 目录

1. [实验总览与 Tier-2 结论](#1-实验总览与-tier-2-结论)
2. [实验过程与时间线](#2-实验过程与时间线)
3. [实验环境与资产索引](#3-实验环境与资产索引)
4. [TensorBoard 访问](#4-tensorboard-访问)
5. [分阶段实验设计与结果](#5-分阶段实验设计与结果)
6. [融合策略消融（P2）](#6-融合策略消融p2)
7. [配方消融（P3）](#7-配方消融p3)
8. [模态消融（P4）](#8-模态消融p4)
9. [Close-out 并行重训](#9-close-out-并行重训)
10. [中文微调轨（M3_M7 → chinese_agent v1/v2）](#10-中文微调轨m3_m7--chinese_agent-v1v2)
11. [结果分析、消融总评与论文叙事](#11-结果分析消融总评与论文叙事)
12. [emotion-agent 前端推理模型选择矩阵](#12-emotion-agent-前端推理模型选择矩阵)
13. [全量 run Best/Last 指标表（磁盘复核）](#13-全量-run-bestlast-指标表磁盘复核)
14. [Checkpoint 与日志清单](#14-checkpoint-与日志清单)
15. [修订记录](#15-修订记录)

---

## 1. 实验总览与 Tier-2 结论

R4（Round 4）为 SDAVT v3 **论文主轨**实验线，共 **6 个阶段 + Close-out 重训 + 中文微调轨**：

| 阶段 | 目的 | Job 数 | 日志桶 |
|------|------|--------|--------|
| P0 | 融合模块修复验证 | 6 | `logs_sdavt_v3_r4/` |
| P1 | 三数据集 ES 主基线 | 3 | 同上 |
| P2 | 五融合策略对比 | 14 | 同上 |
| P3-C | CREMA Acc 配方消融 | 3 (+3 close-out) | 同上 |
| P3-M | MELD F1 配方消融 | 8 | 同上 |
| P4 | 模态组合消融 Table 4 | 21 | 同上 |
| **队列合计** | | **55 done** | |
| **P5 中文微调** | MELD 中文 BERT（v1→v2 全量） | **2**（+smoke 归档） | 同上；§10 |

当前 `logs_sdavt_v3_r4/` 含 **metrics.csv 的 run ≈ 59**（队列正式 run + close-out + 中文微调等）。

### Tier-2 验收（论文口径，2026-07-09 close-out）

| 数据集 | 指标目标 | **最终 Champion** | Best F1 | Best Acc | 判定 |
|--------|----------|-------------------|---------|----------|------|
| **MELD** | F1≥0.59, Acc≥0.62 | **M3_M7_combo** | **0.6957** @ ep31 | **0.7121** @ ep31 | **PASS** |
| **MOSEI** | F1≥0.67 | **F_O_ES**（P2 融合冠军） | **0.6792** @ ep12 | 0.7269 @ ep12 | **PASS** |
| **CREMA** | Acc≥0.63 | **C4_C3** warm-start | 0.6057 @ ep65 | **0.6048** @ ep65 | **CLOSE-OUT**（差 ≈2.5pp） |

### 中文部署轨验收（2026-07-15，相对 v1）

| Run | Best val F1 | Best Acc | 相对前一档 | Agent Preset |
|-----|-------------|----------|------------|--------------|
| M3_M7_combo（英文参照） | 0.6957 | 0.7121 | — | `sdavt_meld_v3_r4` |
| chinese_agent **v1** | 0.6010 @ ep9 | 0.6273 @ep9（max Acc 0.6336 @13） | vs M3_M7 **−9.5pp** | `sdavt_meld_zh_agent` |
| chinese_agent **v2 全量** | **0.6114** @ ep5 | **0.6363** | vs v1 **+1.0pp** | **`sdavt_meld_zh_agent_v2`（默认）** |

**Audit：** 队列 tier2_fail=1（CREMA 未达 0.63）；中文轨以**在线可用性**验收，**勿与英文 M3_M7 混排名**。

### 1.1 Best vs Last 稳定性（论文必读）

从 §13 全表可见：**多数 run 末轮低于峰值**，部署与论文制表一律以 **Best F1（或 CREMA 的 Best Acc）checkpoint** 为准。

| 代表性 Run | Best F1 | Last F1 | Δ (Last−Best) | 解读 |
|------------|---------|---------|---------------|------|
| M3_M7_combo | 0.6957 | 0.6880 | −0.008 | 稳定；适合英文主表 |
| F_M_ES | 0.6109 | 0.4522 | **−0.159** | 早峰后大跌；**必须用 best ckpt** |
| M3_M0_baseline | 0.6080 | 0.4524 | −0.156 | 同上 |
| F_O_ES | 0.6792 | 0.6746 | −0.005 | 极稳 |
| C4_C3 warmstart | 0.6057 | 0.5574 | −0.048 | 中等退化 |
| chinese_v2（全量段） | 0.6114 | 0.6020 | −0.009 | 早停恢复 best；稳定 |

---

## 2. 实验过程与时间线

| 时间段 | 阶段 | 关键事件 |
|--------|------|----------|
| 2026-06-22 ~ 06-23 | P0 | 融合修复 sanity（STD/TS/LFT 等） |
| 2026-06-23 | P1 | 三数据集 ES 主基线 R4_B_* 完成 |
| 2026-06-23 ~ 06-24 | P2 | 14 组融合对比；**ES 锁定**为 P3/P4 默认 |
| 2026-06-25 ~ 06-26 | P3 | MELD 8 组配方 + CREMA 3 组配方并行；**M3_M7 F1=0.6957** |
| 2026-06-26 ~ 06-30 | P4 | 21 组模态消融（Table 4） |
| 2026-07-07 ~ 07-09 | Close-out | C4_C3 warm-start Acc=0.605；R4_A_M_V 重训；队列 GPU 线关闭 |
| 2026-07 上旬 | **P5-v1** | `M3_M7_chinese_agent`：文本→**bert-base-chinese**；Best F1=**0.6010** |
| 2026-07-13 ~ 07-15 | **P5-v2** | 中文 ASR 伪标签 + agent_capture 注入；smoke(256)→**全量**；Best F1=**0.6114** @ep5，early-stop @ep10 |
| 2026-07-16 | 部署 | `.env` 默认 `MODEL_CHECKPOINT_PRESET=sdavt_meld_zh_agent_v2`；语言自动路由 zh→v2 / en→M3_M7 |

**执行方式：** 双 GPU 队列 `sdavt_r4_worker.sh` + tmux；中文微调脚本 `scripts/finetune_m3m7_chinese_agent_v2.sh`；指标写入 `metrics.csv` / TensorBoard。

### 2.1 统一训练设置（R4 主轨典型值）

| 项目 | 设置 |
|------|------|
| 模型 | `MultimodalEmotionModel` + YAML 配置驱动 |
| 优化器 | AdamW，lr=1e-4（finetune 可降），weight_decay=0.01 |
| 调度 | cosine + warmup；gradient_clip=1.0 |
| Batch | MELD/CREMA 常 batch=1 + grad_accum=2；MOSEI 视显存调整 |
| Early stopping | monitor=val_f1 或 val_acc，patience 6~10 |
| 视频 | ResNet50，112×112，4 frames，3.0s 窗口 |
| 音频 | wav2vec2-base（P1/P2/P4）或 **large**（P3 冠军配方） |
| 文本 | bert-base-uncased（P1/P4）或 **roberta-base**（M3_M7） |
| MOSEI 视频 | OpenFace2 npy 713-d 时序特征（`input_type: npy`） |
| 融合 | P2 后固定 **emotion_shift**，leader 默认 text（P4 单模态 yaml 内改 leader_modal） |
| 隔离目录 | `checkpoints_sdavt_v3_r4/`、`logs_sdavt_v3_r4/`、`outputs_sdavt_v3_r4/` |

---

## 3. 实验环境与资产索引

| 资源 | 路径 | 数量 / 说明 |
|------|------|-------------|
| 训练日志 | `logs_sdavt_v3_r4/` | **≈59** 含 `metrics.csv` 的 run（含中文 v1/v2） |
| Checkpoint | `checkpoints_sdavt_v3_r4/` | 含 M3_M7、C4_C3、**chinese_agent / v2** 等 |
| 队列状态 | `outputs_sdavt_v3_r4/experiment_queue.json` | 55 jobs |
| 中文微调日志 | `outputs_sdavt_v3_r4/status/m3m7_zh_*.log` | v1 / v2 smoke / **v2 full**（`EXIT_CODE=0`） |
| Close-out 快照 | `outputs_sdavt_v3_r4/status/r4_closeout_snapshot_20260709.json` | — |
| Agent Preset | `emotion-agent/backend/app/core/config.py` | `CHECKPOINT_PRESETS` + `PRESET_METADATA` |

**统一模型骨架：** `MultimodalEmotionModel`（ResNet50 / Wav2Vec2 / BERT-RoBERTa + 可切换融合）  
**默认融合（P2 后）：** `emotion_shift`（CFN-ESA 风格 Emotion-Shift + cross-attn）  
**训练入口：** `scripts/train.py`；**队列 worker：** `scripts/sdavt_r4_worker.sh`

---

## 4. TensorBoard 访问

所有 R4 训练曲线（loss / F1 / Acc / 分 loss 项）写入各 run 的 TensorBoard event 文件，logdir 统一为：

```text
project/logs_sdavt_v3_r4/
```

| 访问方式 | URL / 命令 |
|----------|------------|
| **本机浏览器** | [http://127.0.0.1:6008](http://127.0.0.1:6008) |
| **局域网 / SSH 转发** | `http://127.0.1.1:6008`（host=127.0.1.1） |
| 启动命令 | `bash scripts/tensorboard_sdavt_r4.sh 6008` |
| tmux 会话 | `tmux attach -t sdavt_r4_tensorboard` |

**查看全部训练图像：**
1. 浏览器打开 TensorBoard → 左侧 **Scalars** 可筛选 `train/`、`val/` 指标
2. 使用右上角 **Filter runs** 搜索 job id，如 `M3_M7`、`C4_C3`、`R4_A_M_V`
3. 每个 run 目录名 = 下表 `Run Dir` 列，与 TB 中 tag 前缀一致

> 端口 **6008** 专用于 R4；旧轨 AP/SDAVT v3 使用 **6007**，互不干扰。

---

## 5. 分阶段实验设计与结果

### p0_fix — P0 融合修复验证（非 ES 融合 sanity）

| Job | Dataset | 模态 | 融合/骨干 | Best F1 | Best Acc | Ep | Collapse | Run Dir | CKPT |
|-----|---------|------|-----------|---------|----------|-----|----------|---------|------|
| F_C_TS | crema | T+A+V | T:bert-base-uncased; A:wav2vec2-base; V:resnet50 | 0.2579 @ ep15 | 0.2836 @ ep27 | 28 | ✓ | `SDAVT_R4_F_C_TS_20260622_140751` | ✓ |
| F_M_STD | meld | T+A+V | T:bert-base-uncased; A:wav2vec2-base; V:resnet50 | 0.4595 @ ep4 | 0.4991 @ ep2 | 13 | ✓ | `SDAVT_R4_F_M_STD_20260622_140751` | ✓ |
| F_M_TS | meld | T+A+V | T:bert-base-uncased; A:wav2vec2-base; V:resnet50 | 0.3826 @ ep9 | 0.4368 @ ep9 | 18 | ✓ | `SDAVT_R4_F_M_TS_20260622_163603` | ✓ |
| F_O_LFT | mosei | T+A+V | T:bert-base-uncased; A:wav2vec2-base; V:resnet50 | 0.5980 @ ep7 | 0.7162 @ ep0 | 9 | ✓ | `SDAVT_R4_F_O_LFT_20260622_214330` | ✓ |
| F_O_STD | mosei | T+A+V | T:bert-base-uncased; A:wav2vec2-base; V:resnet50 | 0.5988 @ ep7 | 0.7162 @ ep0 | 9 | ✓ | `SDAVT_R4_F_O_STD_20260622_222125` | ✓ |
| F_O_TS | mosei | T+A+V | T:bert-base-uncased; A:wav2vec2-base; V:resnet50 | 0.5978 @ ep0 | 0.7162 @ ep0 | 9 | ✓ | `SDAVT_R4_F_O_TS_20260622_232044` | ✓ |

### p1_baseline — P1 单域 AVT + emotion_shift 主基线

| Job | Dataset | 模态 | 融合/骨干 | Best F1 | Best Acc | Ep | Collapse | Run Dir | CKPT |
|-----|---------|------|-----------|---------|----------|-----|----------|---------|------|
| R4_B_C0 | crema | T+A+V | T:bert-base-uncased; A:wav2vec2-base; V:resnet50 | 0.5889 @ ep19 | 0.5874 @ ep19 | 32 | ✓ | `SDAVT_R4_R4_B_C0_20260623_005627` | ✓ |
| R4_B_M1 | meld | T+A+V | T:bert-base-uncased; A:wav2vec2-base; V:resnet50 | 0.5680 @ ep3 | 0.5966 @ ep2 | 14 | ✓ | `SDAVT_R4_R4_B_M1_20260623_005627` | ✓ |
| R4_B_O0 | mosei | T+A+V | T:bert-base-uncased; A:wav2vec2-base; V:resnet50 | 0.6792 @ ep12 | 0.7269 @ ep9 | 21 | ✓ | `SDAVT_R4_R4_B_O0_20260623_032138` | ✓ |

### p2_fusion — P2 五融合策略对比（选定 ES 为后续主融合）

| Job | Dataset | 模态 | 融合/骨干 | Best F1 | Best Acc | Ep | Collapse | Run Dir | CKPT |
|-----|---------|------|-----------|---------|----------|-----|----------|---------|------|
| F_C_ES | crema | T+A+V | T:bert-base-uncased; A:wav2vec2-base; V:resnet50 | 0.5786 @ ep28 | 0.5860 @ ep41 | 41 | ✓ | `SDAVT_R4_F_C_ES` | ✓ |
| F_C_LFA | crema | T+A+V | T:bert-base-uncased; A:wav2vec2-base; V:resnet50 | 0.2978 @ ep26 | 0.3065 @ ep15 | 28 | ✓ | `SDAVT_R4_F_C_LFA_20260624_070022` | ✓ |
| F_C_LFT | crema | T+A+V | T:bert-base-uncased; A:wav2vec2-base; V:resnet50 | 0.3081 @ ep15 | 0.3320 @ ep15 | 28 | ✓ | `SDAVT_R4_F_C_LFT_20260624_073903` | ✓ |
| F_C_STD | crema | T+A+V | T:bert-base-uncased; A:wav2vec2-base; V:resnet50 | 0.2405 @ ep14 | 0.2594 @ ep4 | 27 | ✓ | `SDAVT_R4_F_C_STD_20260624_081739` | ✓ |
| F_C_TS | crema | T+A+V | T:bert-base-uncased; A:wav2vec2-base; V:resnet50 | 0.2436 @ ep29 | 0.3051 @ ep27 | 42 | ✓ | `SDAVT_R4_F_C_TS_20260624_085905` | ✓ |
| F_M_ES | meld | T+A+V | T:bert-base-uncased; A:wav2vec2-base; V:resnet50 | 0.6109 @ ep3 | 0.6245 @ ep3 | 16 | ✓ | `SDAVT_R4_F_M_ES` | ✓ |
| F_M_LFA | meld | T+A+V | T:bert-base-uncased; A:wav2vec2-base; V:resnet50 | 0.4562 @ ep3 | 0.4937 @ ep1 | 12 | ✓ | `SDAVT_R4_F_M_LFA_20260623_210443` | ✓ |
| F_M_LFT | meld | T+A+V | T:bert-base-uncased; A:wav2vec2-base; V:resnet50 | 0.4207 @ ep5 | 0.4486 @ ep9 | 14 | ✓ | `SDAVT_R4_F_M_LFT_20260623_225301` | ✓ |
| F_M_STD | meld | T+A+V | T:bert-base-uncased; A:wav2vec2-base; V:resnet50 | 0.4447 @ ep5 | 0.4973 @ ep2 | 14 | ✓ | `SDAVT_R4_F_M_STD_20260624_005729` | ✓ |
| F_M_TS | meld | T+A+V | T:bert-base-uncased; A:wav2vec2-base; V:resnet50 | 0.3682 @ ep12 | 0.4314 @ ep5 | 21 | ✓ | `SDAVT_R4_F_M_TS_20260624_030111` | ✓ |
| F_O_ES | mosei | T+A+V | T:bert-base-uncased; A:wav2vec2-base; V:resnet50 | 0.6792 @ ep12 | 0.7269 @ ep9 | 21 | ✓ | `SDAVT_R4_F_O_ES_20260624_101647` | ✓ |
| F_O_LFT | mosei | T+A+V | T:bert-base-uncased; A:wav2vec2-base; V:resnet50 | 0.5980 @ ep7 | 0.7162 @ ep0 | 9 | ✓ | `SDAVT_R4_F_O_LFT_20260624_110504` | ✓ |
| F_O_STD | mosei | T+A+V | T:bert-base-uncased; A:wav2vec2-base; V:resnet50 | 0.5988 @ ep7 | 0.7162 @ ep0 | 9 | ✓ | `SDAVT_R4_F_O_STD_20260624_115119` | ✓ |
| F_O_TS | mosei | T+A+V | T:bert-base-uncased; A:wav2vec2-base; V:resnet50 | 0.5978 @ ep0 | 0.7162 @ ep0 | 9 | ✓ | `SDAVT_R4_F_O_TS_20260624_123518` | ✓ |

### p3_c3 — P3-C CREMA 配方消融（Acc 导向）

| Job | Dataset | 模态 | 融合/骨干 | Best F1 | Best Acc | Ep | Collapse | Run Dir | CKPT |
|-----|---------|------|-----------|---------|----------|-----|----------|---------|------|
| C3_C1_baseline | crema | T+A+V | T:bert-base-uncased; A:wav2vec2-base; V:resnet50 | 0.5336 @ ep29 | 0.5417 @ ep29 | 45 | ✓ | `SDAVT_R4_C3_C1_baseline_20260625_200937` | ✓ |
| C3_C2_w2v_large | crema | T+A+V | T:bert-base-uncased; A:wav2vec2-large-960h; V:re | 0.5629 @ ep31 | 0.5672 @ ep31 | 47 | ✓ | `SDAVT_R4_C3_C2_w2v_large_20260626_004150` | ✓ |
| C3_C3_focal | crema | T+A+V | T:bert-base-uncased; A:wav2vec2-base; V:resnet50 | 0.5526 @ ep41 | 0.5565 @ ep41 | 50 | ✓ | `SDAVT_R4_C3_C3_focal_20260626_043125` | ✓ |

### p3_m3 — P3-M MELD 配方消融（F1 导向）

| Job | Dataset | 模态 | 融合/骨干 | Best F1 | Best Acc | Ep | Collapse | Run Dir | CKPT |
|-----|---------|------|-----------|---------|----------|-----|----------|---------|------|
| M3_M0_baseline | meld | T+A+V | T:bert-base-uncased; A:wav2vec2-base; V:resnet50 | 0.6080 @ ep3 | 0.6218 @ ep3 | 19 | ✓ | `SDAVT_R4_M3_M0_baseline_20260625_200937` | ✓ |
| M3_M1_roberta | meld | T+A+V | T:roberta-base; A:wav2vec2-base; V:resnet50; fus | 0.6823 @ ep17 | 0.6968 @ ep22 | 24 | ✓ | `SDAVT_R4_M3_M1_roberta` | ✓ |
| M3_M2_w2v_large | meld | T+A+V | T:bert-base-uncased; A:wav2vec2-large-960h; V:re | 0.5572 @ ep3 | 0.6020 @ ep3 | 19 | ✓ | `SDAVT_R4_M3_M2_w2v_large_20260625_233919` | ✓ |
| M3_M3_uniform | meld | T+A+V | T:bert-base-uncased; A:wav2vec2-base; V:resnet50 | 0.6105 @ ep3 | 0.6245 @ ep3 | 19 | ✓ | `SDAVT_R4_M3_M3_uniform_20260626_031222` | ✓ |
| M3_M4_focal | meld | T+A+V | T:bert-base-uncased; A:wav2vec2-base; V:resnet50 | 0.6079 @ ep3 | 0.6209 @ ep3 | 19 | ✓ | `SDAVT_R4_M3_M4_focal_20260626_062916` | ✓ |
| M3_M5_context | meld | T+A+V | T:bert-base-uncased; A:wav2vec2-base; V:resnet50 | 0.5725 @ ep3 | 0.5912 @ ep5 | 19 | ✓ | `SDAVT_R4_M3_M5_context_20260626_073046` | ✓ |
| M3_M6_moddrop | meld | T+A+V | T:bert-base-uncased; A:wav2vec2-base; V:resnet50 | 0.6079 @ ep6 | 0.6245 @ ep6 | 22 | ✓ | `SDAVT_R4_M3_M6_moddrop_20260626_113831` | ✓ |
| M3_M7_combo | meld | T+A+V | T:roberta-base; A:wav2vec2-large-960h; V:resnet5 | 0.6957 @ ep31 | 0.7121 @ ep31 | 33 | ✓ | `SDAVT_R4_M3_M7_combo` | ✓ |

### p4_modal — P4 模态消融（7 种模态组合 × 3 数据集）

| Job | Dataset | 模态 | 融合/骨干 | Best F1 | Best Acc | Ep | Collapse | Run Dir | CKPT |
|-----|---------|------|-----------|---------|----------|-----|----------|---------|------|
| R4_A_C_A | crema | A | A:wav2vec2-base; fusion=emotion_shift; leader=au | 0.1412 @ ep30 | 0.1909 @ ep33 | 43 | ✓ | `SDAVT_R4_R4_A_C_A` | ✓ |
| R4_A_C_AT | crema | T+A | T:bert-base-uncased; A:wav2vec2-base; fusion=emo | 0.1312 @ ep16 | 0.1815 @ ep11 | 29 | ✓ | `SDAVT_R4_R4_A_C_AT` | ✓ |
| R4_A_C_AV | crema | A+V | A:wav2vec2-base; V:resnet50; fusion=emotion_shif | 0.3303 @ ep32 | 0.3562 @ ep41 | 45 | ✓ | `SDAVT_R4_R4_A_C_AV` | ✓ |
| R4_A_C_AVT | crema | T+A+V | T:bert-base-uncased; A:wav2vec2-base; V:resnet50 | 0.3263 @ ep36 | 0.3575 @ ep36 | 49 | ✓ | `SDAVT_R4_R4_A_C_AVT` | ✓ |
| R4_A_C_T | crema | T | T:bert-base-uncased; fusion=emotion_shift; leade | 0.0891 @ ep8 | 0.1761 @ ep8 | 21 | ✓ | `SDAVT_R4_R4_A_C_T` | ✓ |
| R4_A_C_V | crema | V | V:resnet50; fusion=emotion_shift; leader=video | 0.3538 @ ep37 | 0.3804 @ ep25 | 50 | ✓ | `SDAVT_R4_R4_A_C_V` | ✓ |
| R4_A_C_VT | crema | T+V | T:bert-base-uncased; V:resnet50; fusion=emotion_ | 0.3159 @ ep21 | 0.3481 @ ep21 | 24 | ✓ | `SDAVT_R4_R4_A_C_VT` | ✓ |
| R4_A_M_A | meld | A | A:wav2vec2-base; fusion=emotion_shift; leader=te | 0.4821 @ ep33 | 0.5072 @ ep12 | 49 | ✓ | `SDAVT_R4_R4_A_M_A` | ✓ |
| R4_A_M_AT | meld | T+A | T:bert-base-uncased; A:wav2vec2-base; fusion=emo | 0.6736 @ ep5 | 0.6913 @ ep5 | 11 | ✓ | `SDAVT_R4_R4_A_M_AT` | ✓ |
| R4_A_M_AV | meld | A+V | A:wav2vec2-base; V:resnet50; fusion=emotion_shif | 0.4780 @ ep21 | 0.5054 @ ep5 | 37 | ✓ | `SDAVT_R4_R4_A_M_AV` | ✓ |
| R4_A_M_AVT | meld | T+A+V | T:bert-base-uncased; A:wav2vec2-base; V:resnet50 | 0.6818 @ ep5 | 0.6958 @ ep5 | 21 | ✓ | `SDAVT_R4_R4_A_M_AVT` | ✓ |
| R4_A_M_T | meld | T | T:bert-base-uncased; fusion=emotion_shift; leade | 0.6741 @ ep6 | 0.6895 @ ep6 | 22 | ✓ | `SDAVT_R4_R4_A_M_T` | ✓ |
| R4_A_M_V | meld | V | V:resnet50; fusion=emotion_shift; leader=video | 0.2690 @ ep9 | 0.4233 @ ep0 | 25 | ✓ | `SDAVT_R4_R4_A_M_V` | ✓ |
| R4_A_M_VT | meld | T+V | T:bert-base-uncased; V:resnet50; fusion=emotion_ | 0.6738 @ ep6 | 0.6895 @ ep6 | 22 | ✓ | `SDAVT_R4_R4_A_M_VT` | ✓ |
| R4_A_O_A | mosei | A | A:wav2vec2-base; fusion=emotion_shift; leader=au | 0.6362 @ ep13 | 0.7162 @ ep0 | 22 | ✓ | `SDAVT_R4_R4_A_O_A` | ✓ |
| R4_A_O_AT | mosei | T+A | T:bert-base-uncased; A:wav2vec2-base; fusion=emo | 0.6922 @ ep6 | 0.7376 @ ep12 | 15 | ✓ | `SDAVT_R4_R4_A_O_AT` | ✓ |
| R4_A_O_AV | mosei | A+V | A:wav2vec2-base; V:resnet50/npy; fusion=emotion_ | 0.6415 @ ep2 | 0.7162 @ ep1 | 11 | ✓ | `SDAVT_R4_R4_A_O_AV` | ✓ |
| R4_A_O_AVT | mosei | T+A+V | T:bert-base-uncased; A:wav2vec2-base; V:resnet50 | 0.6982 @ ep11 | 0.7338 @ ep7 | 15 | ✓ | `SDAVT_R4_R4_A_O_AVT` | ✓ |
| R4_A_O_T | mosei | T | T:bert-base-uncased; fusion=emotion_shift; leade | 0.7087 @ ep15 | 0.7483 @ ep22 | 24 | ✓ | `SDAVT_R4_R4_A_O_T` | ✓ |
| R4_A_O_V | mosei | V | V:resnet50/npy; fusion=emotion_shift; leader=vid | 0.6274 @ ep0 | 0.7162 @ ep1 | 9 | ✓ | `SDAVT_R4_R4_A_O_V` | ✓ |
| R4_A_O_VT | mosei | T+V | T:bert-base-uncased; V:resnet50/npy; fusion=emot | 0.7050 @ ep19 | 0.7483 @ ep26 | 28 | ✓ | `SDAVT_R4_R4_A_O_VT` | ✓ |

### p3_c_plus — Close-out CREMA 加码

| Job | Dataset | 模态 | 融合/骨干 | Best F1 | Best Acc | Ep | Collapse | Run Dir | CKPT |
|-----|---------|------|-----------|---------|----------|-----|----------|---------|------|
| C4_C1_combo_acc | crema | T+A+V | 激进 combo | — | — | — | **failed_nan** | 已迁至 `logs_sdavt_v3_r4_archived/SDAVT_R4_C4_C1_combo_acc` | — |
| C4_C2_c3_base_acc | crema | T+A+V | w2v-large 等 | 0.3152 @ ep48 | 0.3535 @ ep51 | 60 | ✓ | `SDAVT_R4_C4_C2_c3_base_acc` | ✓ |
| C4_C3_c3_warmstart_acc | crema | T+A+V | w2v-large + warm-start | **0.6057 @ ep65** | **0.6048 @ ep65** | 48 val（ep 至 79） | ✓ | `SDAVT_R4_C4_C3_c3_warmstart_acc` | ✓ |

### p5_chinese — 中文微调（接在 M3_M7 之后，详表见 §10）

| Job | Dataset | 文本骨干 | Best F1 | Best Acc | Ep | Run Dir | CKPT | Preset |
|-----|---------|----------|---------|----------|-----|---------|------|--------|
| M3_M7_chinese_agent | meld | **bert-base-chinese** | **0.6010 @ ep9** | 0.6273 | 14 val | `SDAVT_R4_M3_M7_chinese_agent` | `checkpoint_finetune_best_f1.pth` | `sdavt_meld_zh_agent` |
| M3_M7_chinese_agent_v2 | meld | **bert-base-chinese** + 中文增强 | **0.6114 @ ep5** | **0.6363** | 23 val / early-stop@10 | `SDAVT_R4_M3_M7_chinese_agent_v2` | `checkpoint_finetune_best_f1.pth` | **`sdavt_meld_zh_agent_v2`** |

Smoke 归档：`checkpoints_sdavt_v3_r4/SDAVT_R4_M3_M7_chinese_agent_v2_smoke_256/`（256 samples，非正式主表）。

---

## 6. 融合策略消融（P2）

P2 在三数据集上对比 **5 种融合**：`standard` / `two_stage` / `leader_follower`（LFA/LFT）/ `emotion_shift`。
**结论：** `emotion_shift`（ES）在三数据集均为 P2 冠军，故 P3/P4/P5 全部固定 ES。

| Dataset | P2 冠军 | Best F1 | vs STD ΔF1 | vs 次优 | Run |
|---------|---------|---------|------------|---------|-----|
| crema | **F_C_ES** | 0.5786 | **+0.338** | vs LFT +0.270 | `SDAVT_R4_F_C_ES` |
| meld | **F_M_ES** | 0.6109 | **+0.166** | vs LFA +0.155 | `SDAVT_R4_F_M_ES` |
| mosei | **F_O_ES** | 0.6792 | **+0.080** | vs STD +0.080 | `SDAVT_R4_F_O_ES_20260624_101647` |

**效果评价：** CREMA 上非 ES 融合接近崩溃（STD/TS≈0.24），说明 **融合选择对表演语料几乎是开关级差异**；MELD 上 ES 相对 STD 仍有 **+16.6pp**；MOSEI 增益较小但仍稳定领先。与早期 `logs_rerun`「ES ≫ standard」一致，且单域协议更干净。

**分融合明细（同数据集内对比）：**

#### CREMA

| Fusion | Job | Best F1 | Best Acc |
|--------|-----|---------|----------|
| emotion_shift | F_C_ES | 0.5786 @ ep28 | 0.5860 @ ep41 |
| leader_follower | F_C_LFT | 0.3081 @ ep15 | 0.3320 |
| leader_follower | F_C_LFA | 0.2978 @ ep26 | 0.3065 |
| two_stage | F_C_TS | 0.2436 @ ep29 | 0.3051 |
| standard | F_C_STD | 0.2405 @ ep14 | 0.2594 |

#### MELD

| Fusion | Job | Best F1 | Best Acc |
|--------|-----|---------|----------|
| emotion_shift | F_M_ES | 0.6109 @ ep3 | 0.6245 |
| leader_follower | F_M_LFA | 0.4562 @ ep3 | 0.4937 |
| standard | F_M_STD | 0.4447 @ ep5 | 0.4973 |
| leader_follower | F_M_LFT | 0.4207 @ ep5 | 0.4486 |
| two_stage | F_M_TS | 0.3682 @ ep12 | 0.4314 |

#### MOSEI

| Fusion | Job | Best F1 | Best Acc |
|--------|-----|---------|----------|
| emotion_shift | F_O_ES | 0.6792 @ ep12 | 0.7269 |
| standard | F_O_STD | 0.5988 @ ep7 | 0.7162 |
| leader_follower | F_O_LFT | 0.5980 @ ep7 | 0.7162 |
| two_stage | F_O_TS | 0.5978 @ ep0 | 0.7162 |

---

## 7. 配方消融（P3）

### 7.1 MELD（P3-M）— 目标 F1≥0.59

| Job | 配方要点 | Best F1 | Best Acc | Δ vs M0 | 效果评价 |
|-----|----------|---------|----------|---------|----------|
| **M3_M7_combo** | RoBERTa + w2v-large + context + moddrop + focal | **0.6957** @31 | **0.7121** | **+0.0877** | **冠军；英文 Agent / 论文主表** |
| **M3_M1_roberta** | 仅换 RoBERTa | **0.6823** @17 | 0.6968 | **+0.0743** | 文本骨干贡献最大单因素 |
| M3_M3_uniform | 均匀采样 | 0.6105 @3 | 0.6245 | +0.0025 | 边际 |
| M3_M0_baseline | ES 基线 | 0.6080 @3 | 0.6218 | 0 | 锚点 |
| M3_M4_focal | focal | 0.6079 @3 | 0.6209 | −0.0001 | 单独无效 |
| M3_M6_moddrop | modality dropout | 0.6079 @6 | 0.6245 | −0.0001 | 单独无效；与 combo 叠加有用 |
| M3_M5_context | context window | 0.5725 @3 | 0.5912 | −0.0355 | 单独有害/不稳定 |
| M3_M2_w2v_large | 仅换 audio large | 0.5572 @3 | 0.6020 | −0.0508 | 单独有害；需与 RoBERTa 组合 |

**M3_M7_combo 配方（冠军）：** `roberta-base` + `wav2vec2-large-960h` + ResNet50；`use_context_window=true`；`modality_dropout=0.1`；focal + label smoothing；dropout 0.35。

**消融解读：** MELD 上 **文本骨干（RoBERTa）是主增益源（+7.4pp）**；音频 large / context / moddrop / focal **单独**几乎无益甚至负增益，但 **组合进 M7** 再抬 **+1.3pp**（相对 M1），说明配方存在协同而非简单叠加。

### 7.2 CREMA（P3-C + P3-C+）— 目标 Acc≥0.63

| Job | 配方要点 | Best Acc | Best F1 | Δ vs C1 | 判定 |
|-----|----------|----------|---------|---------|------|
| **C4_C3_c3_warmstart_acc** | warm-start + Acc 导向 | **0.6048** @65 | 0.6057 | +0.063 | **CLOSE-OUT champion**（仍差 Tier-2 0.63） |
| C3_C2_w2v_large | w2v-large | 0.5672 @31 | 0.5629 | +0.026 | 队列最优；→ preset `sdavt_crema_r4` |
| C3_C3_focal | focal | 0.5565 @41 | 0.5526 | +0.015 | 弱于 C2 |
| C3_C1_baseline | 基线 | 0.5417 @29 | 0.5336 | 0 | 锚点 |
| C4_C2_c3_base_acc | 激进 recipe | 0.3535 @51 | 0.3152 | −0.19 | **退化** |
| C4_C1_combo_acc | 激进 combo | — | — | — | **failed_nan** |

**消融解读：** CREMA 上 **稳健加 w2v-large（C2）有效**；**warm-start（C4_C3）再抬 ~3.8pp Acc**；盲目换 recipe（C4_C1/C2）会崩溃。论文应报告 C4_C3 为尽力 close-out，并说明未达 0.63 的域难度（表演语料 + 文本弱）。

---

## 8. 模态消融（P4）

P4 在 ES 融合固定下，对 **7 种模态组合**（A/T/V/AT/AV/VT/AVT）× **3 数据集** 进行消融。
Job 命名：`R4_A_{C|M|O}_{modality}`（C=CREMA, M=MELD, O=MOSEI）。

### MELD

| 模态 | Job | Best F1 | Best Acc | Δ vs AVT | 分析 |
|------|-----|---------|----------|----------|------|
| T+A+V | R4_A_M_AVT | **0.6818** @5 | 0.6958 | 0 | 全模态锚点（BERT 骨干，非 M7） |
| T | R4_A_M_T | 0.6741 @6 | 0.6895 | −0.008 | **文本单模态≈全模态** → 对话情感文本主导 |
| T+V | R4_A_M_VT | 0.6738 @6 | 0.6895 | −0.008 | 加 V 几乎无增益增益 |
| T+A | R4_A_M_AT | 0.6736 @5 | 0.6913 | −0.008 | 同上 |
| A | R4_A_M_A | 0.4821 @33 | 0.5072 | −0.200 | 纯音频中等 |
| A+V | R4_A_M_AV | 0.4780 @21 | 0.5054 | −0.204 | 无文本时上限受限 |
| V | R4_A_M_V | 0.2690 @9 | 0.4233 | −0.413 | **视频单模态极弱**（任务固有，非训练 bug） |

### MOSEI

| 模态 | Job | Best F1 | Best Acc | Δ vs AVT | 分析 |
|------|-----|---------|----------|----------|------|
| T | R4_A_O_T | **0.7087** @15 | **0.7483** | **+0.011** | **纯文本反超 AVT** → 字幕/转写极强 |
| T+V | R4_A_O_VT | 0.7050 @19 | 0.7483 | +0.007 | 接近 T |
| T+A+V | R4_A_O_AVT | 0.6982 @11 | 0.7338 | 0 | 全模态略低于 T（融合噪声可能） |
| T+A | R4_A_O_AT | 0.6922 @6 | 0.7376 | −0.006 | |
| A+V | R4_A_O_AV | 0.6415 @2 | 0.7162 | −0.057 | |
| A | R4_A_O_A | 0.6362 @13 | 0.7162 | −0.062 | |
| V | R4_A_O_V | 0.6274 @0 | 0.7162 | −0.071 | 仍明显高于 CREMA/MELD 的 V-only |

### CREMA

| 模态 | Job | Best F1 | Best Acc | Δ vs AVT | 分析 |
|------|-----|---------|----------|----------|------|
| V | R4_A_C_V | **0.3538** @37 | **0.3804** | **+0.028** | **视觉最强**（表演表情） |
| A+V | R4_A_C_AV | 0.3303 @32 | 0.3562 | +0.004 | |
| T+A+V | R4_A_C_AVT | 0.3263 @36 | 0.3575 | 0 | 加文本未超 V |
| T+V | R4_A_C_VT | 0.3159 @21 | 0.3481 | −0.010 | |
| A | R4_A_C_A | 0.1412 @30 | 0.1909 | −0.185 | collapse 量级 |
| T+A | R4_A_C_AT | 0.1312 @16 | 0.1815 | −0.195 | collapse |
| T | R4_A_C_T | 0.0891 @8 | 0.1761 | −0.237 | **文本几乎无信息**（固定句脚本） |

**跨数据集模态结论（支撑 Agent 策略）：**

| 场景 | 应依赖 | 应降权 / 跳过 | 对应部署 |
|------|--------|---------------|----------|
| 英文对话（MELD 类） | 文本 ≫ 音频 > 视频 | 勿仅靠视频 | `sdavt_meld_v3_r4` |
| 中文对话 Agent | 文本（中文 BERT）+ 音频 | 弱 ASR 时降文本权重 | `sdavt_meld_zh_agent_v2` |
| CREMA / 表演式 | 视频+音频 | **勿依赖文本** | `sdavt_crema_r4` + leader_audio / skip_text |
| MOSEI | 文本主导 | AVT 可选 | `sdavt_mosei_r4`（实验） |

---

## 9. Close-out 并行重训

| Job | 完成时间 | Best 指标 | 判定 | Run |
|-----|----------|-----------|------|-----|
| **C4_C3** | 2026-07-09T06:32:00+08:00 | F1=0.6057 Acc=0.6048 | PARTIAL（未达 Acc0.63） | `SDAVT_R4_C4_C3_c3_warmstart_acc` |
| **R4_A_M_V** | 2026-07-09T07:55:00+08:00 | F1=0.2690 Acc=0.4233 | FAIL（V-only 下限确认） | `SDAVT_R4_R4_A_M_V` |

---

## 10. 中文微调轨（M3_M7 → chinese_agent v1/v2）

### 10.1 动机与设计原则

1. **问题：** 英文 M3_M7（RoBERTa）在中文 ASR 文本上分词/embedding 错配，在线中文效果差。  
2. **策略：** **仅对 MELD** 做中文 BERT 二阶段微调；**不对 MOSEI/CREMA 做中文微调**（英文语料无中文监督；CREMA 文本本就崩溃，ROI 低）。  
3. **协议：** 离线仍在 MELD val 上报告 F1/Acc，但与英文 M3_M7 **分表**；部署以中文 E2E 为主。

### 10.2 实验配置对照

| 项目 | v1 `chinese_agent` | v2 `chinese_agent_v2`（全量） |
|------|--------------------|-------------------------------|
| Config | `config/sdavt_v3_r4/p3_m3/meld/M3_M7_chinese_agent.yaml` | `.../M3_M7_chinese_agent_v2.yaml` |
| 文本骨干 | **bert-base-chinese** | 同左 |
| 初始化 | 自 M3_M7 英文权重迁移（文本塔重初始化） | **自 v1** `checkpoint_finetune_best_f1.pth`（partial load 1031 tensors） |
| 训练数据 | MELD（英文字幕进中文分词器） | MELD + **≈500×`*_zh.txt` ASR 伪标签** + **≈97 agent_capture** |
| `max_train_samples` | 全量 | **0（全量）**；smoke=256 已归档 |
| 日志子集规模 | — | train **10085** / val **1108**（meld） |
| 状态日志 | `outputs_sdavt_v3_r4/status/m3m7_zh_finetune.log` | `m3m7_zh_v2_full_finetune.log` |
| metrics | `logs_sdavt_v3_r4/SDAVT_R4_M3_M7_chinese_agent/` | `.../chinese_agent_v2/` |
| Checkpoint | `checkpoints_sdavt_v3_r4/SDAVT_R4_M3_M7_chinese_agent/checkpoint_finetune_best_f1.pth` | `.../chinese_agent_v2/checkpoint_finetune_best_f1.pth` |

### 10.3 指标与训练曲线要点

| Run | Best F1 @ep | Acc@BestF1 | Max Acc @ep | Last F1 @ep | 早停 / 备注 |
|-----|-------------|------------|-------------|-------------|-------------|
| M3_M7_combo（英文参照） | **0.6957 @31** | **0.7121** | 同左 | 0.6880 @32 | 论文英文主表 |
| chinese_agent **v1** | **0.6010 @9** | 0.6273 | 0.6336 @13 | 0.5973 @13 | 换中文 BERT ≈−9.5pp |
| chinese_agent **v2 全量** | **0.6114 @5** | **0.6363** | 同左 | 0.6020 @10 | patience=5；restore ep5；`EXIT_CODE=0` |
| chinese_agent **v2 smoke** | 0.6054 @10–11 | 0.6291 | — | — | 256 samples；非正式主表 |

**注意：** `logs_sdavt_v3_r4/SDAVT_R4_M3_M7_chinese_agent_v2/metrics.csv` 内 **先写了 smoke 12 个 val epoch，再接全量 11 个 val epoch**（epoch 序号重置为 0）。主表数字取**全量段** Best F1=0.6114；smoke 权重在 `checkpoints_.../SDAVT_R4_M3_M7_chinese_agent_v2_smoke_256/`。

#### v1 全量 val 曲线（`SDAVT_R4_M3_M7_chinese_agent`）

| Epoch | val_f1 | val_acc | val_loss | 事件 |
|-------|--------|---------|----------|------|
| 0 | 0.5114 | 0.5767 | 0.7308 | 中文 BERT 冷启动 |
| 4 | 0.5726 | 0.6182 | 0.7732 | 快速爬升 |
| 7 | 0.5992 | 0.6273 | 0.7004 | 接近峰值 |
| **9** | **0.6010** | **0.6273** | **0.6625** | **Best F1** |
| 13 | 0.5973 | 0.6336 | 0.7332 | Last；Acc 略高于 Best-F1 轮 |

#### v2 全量段 val 曲线（接在 smoke 之后；日志 `m3m7_zh_v2_full_finetune.log`）

| Epoch | val_f1 | val_acc | 事件 |
|-------|--------|---------|------|
| 0 | 0.6017 | 0.6282 | 自 v1 继承；首次写 best_f1 |
| 3 | 0.6041 | 0.6345 | 更新 best_f1 |
| **5** | **0.6114** | **0.6363** | **最终 best_f1** |
| 6 | 0.6076 | 0.6363 | 未超越 |
| 10 | 0.6020 | 0.6336 | Early stopping；restore ep5 |

#### v2 smoke 段（同 CSV 前 12 行，仅流水线验证）

| 区间 | Best F1 | 说明 |
|------|---------|------|
| ep0–11（smoke） | **0.6054** | `MAX_TRAIN_SAMPLES=256`；已归档 ckpt，**不进论文主表** |

### 10.4 中文消融对比

| 对比 | Δ val F1 | 结论 |
|------|----------|------|
| v1 vs M3_M7（英文） | **−0.0947** | 词表适配换骨干有代价；监督仍偏英文 MELD |
| **v2 vs v1** | **+0.0103** | 中文伪标签 + 采集注入有效 |
| v2 smoke(256) vs 全量 | 全量更稳 | smoke 仅流水线验证，**主表用全量** |
| MOSEI/CREMA 中文微调 | — | **不做**（见 §12.3） |

### 10.5 在线验收（2026-07-16）

- 默认 preset：`sdavt_meld_zh_agent_v2`  
- 冒烟：中文句「我很难过」→ **sad**，conf≈**0.534**  
- 语言自动路由：zh→v2，en→`sdavt_meld_v3_r4`

---

## 11. 结果分析、消融总评与论文叙事

### 11.1 分问题消融总评

| 科学问题 | 关键证据 | 定量结论 | 写作注意 |
|----------|----------|----------|----------|
| 融合策略 | P2 三数据集 | ES 相对 STD：MELD **+0.166** F1；CREMA **+0.338**；MOSEI **+0.080** | 核心方法结论 |
| MELD 配方 | P3-M | RoBERTa **+0.074**；combo **+0.088→0.696** | 论文 Table 主结果 |
| CREMA 配方 | P3-C / C+ | C2→0.567；C4_C3→**0.605**；激进 recipe 崩溃 | 报告尽力值 + 未达 Tier-2 |
| 模态贡献 | P4 | MELD/MOSEI **文本主导**；CREMA **视觉主导、文本崩溃** | 支撑 Agent 分场景策略 |
| 中文适配 | P5 | v1 0.601→v2 **0.611**；相对英文 −9.5pp | **分表**：离线英文 vs 部署中文 |

### 11.2 各组效果一句话评价

| 组 | 效果评价 |
|----|----------|
| P0 | Sanity：非 ES 融合可跑通，数值不作主结论 |
| P1 | 三域 ES 基线；MOSEI 已近达标；MELD/CREMA 待配方抬升 |
| P2 | **ES 锁定成功**；CREMA 上 STD/TS 近崩溃，说明融合选择不可省 |
| P3-M | **M7 夺冠**；文本骨干 > 其它单因素；combo 有协同 |
| P3-C | w2v-large 有效；focal 弱；close-out warm-start 接近但未达 0.63 |
| P4 | 模态依赖强分域；解释「为何中文 Agent 仍要文本、CREMA 可 skip_text」 |
| P5 | 中文轨可用；v2 默认部署；勿与 M7 英文分数直接比高低 |

### 11.3 论文推荐数字

| Table | 内容 | 推荐数值 |
|-------|------|----------|
| 主结果 MELD（英文） | M3_M7_combo | F1=**0.6957**, Acc=**0.7121** |
| 主结果 MOSEI | F_O_ES | F1=**0.6792**, Acc=0.7269 |
| 主结果 CREMA | C4_C3 | Acc=**0.6048**, F1=0.6057 |
| 融合消融 | ES vs STD（MELD） | +**0.166** F1 |
| 模态脚注 | R4_A_M_V | F1≈0.269（V-only 下限） |
| 中文部署 | chinese_agent_v2 | F1=**0.6114**, Acc=**0.6363** |

### 11.4 与早期混合轨的关系（勿混排名）

| 日志桶 | 协议 | 代表峰值 | 用途 |
|--------|------|----------|------|
| `logs_rerun` | 三域混合 | Best Acc≈0.445（ES） | 早期融合证据 |
| `logs_accuracy_seq` | 混合 AP2 | Best Acc≈0.61 | 混合上限讨论 |
| **`logs_sdavt_v3_r4`** | **单域** | MELD F1 **0.696** | **学位论文主表** |

### 11.5 部署选型一句话（对接 §12）

- **中文默认** → `sdavt_meld_zh_agent_v2`（F1=0.6114）  
- **英文 / 论文 MELD** → `sdavt_meld_v3_r4`（F1=0.6957）  
- **MOSEI / CREMA** → 仅实验 preset；**不做中文微调**  

---

## 12. emotion-agent 前端推理模型选择矩阵

> **代码权威源**：`emotion-agent/backend/app/core/config.py` → `CHECKPOINT_PRESETS` + `PRESET_METADATA`。  
> 前端下拉 / `/model/status` 应与本表数字一致；改 checkpoint 时同步改 metadata。

### 12.1 完整 Preset 一览（UI 选项权威参考 · AVT P0–P5）

| 优先级 | Preset ID | UI 分组 | 语言 | Best F1 / Acc | 来源 Job / 日志 | 推荐 | 选用场景 |
|--------|-----------|---------|------|---------------|-----------------|------|----------|
| **P0** | **`sdavt_meld_zh_agent_v2`** | 推荐部署 | zh | **0.6114 / 0.6363** | `SDAVT_R4_M3_M7_chinese_agent_v2` | **默认** | 中文 ASR / Agent |
| **P1** | **`sdavt_meld_v3_r4`** | 推荐部署 | en | **0.6957 / 0.7121** | `SDAVT_R4_M3_M7_combo` | 英文推荐 | 英文对话、论文演示 |
| P2 | `sdavt_meld_zh_agent` | 中文对照 | zh | 0.6010 / 0.6273 | `SDAVT_R4_M3_M7_chinese_agent` | 对照 | 消融：相对 v2 −1.0pp |
| P3 | `sdavt_mosei_r4` | 实验 | en | 0.6792 / 0.7269 | `SDAVT_R4_F_O_ES_20260624_101647` | experimental | MOSEI 单域；**勿中文默认** |
| P4 | `sdavt_crema_r4` | 实验 | — | Acc **0.6048** / F1 0.6057 | `SDAVT_R4_C4_C3_c3_warmstart_acc` | experimental | CREMA Warmstart；leader_audio |
| P5 | `ap2_m1` | 历史 | mixed | F1≈0.56 / Acc≈0.61 | AP2 M1 | — | 三混合演示；勿与 R4 混排名 |

默认下拉**隐藏**：`meld_only` / `mosei_only` / `agent_chinese` / `ap4_w005`（`ui_visible=false`；`/model/status?all=1` 可调试）。

**前端实现：** `/model/status` 透传 `priority` / `display_label` / `group`；`App.jsx` optgroup + Auto 语言路由 + `POST /model/preload`。

### 12.2 自动语言 → Preset

| 检测语言 | `suggested_preset` |
|----------|-------------------|
| 中文 zh | `sdavt_meld_zh_agent_v2` |
| 英文 en | `sdavt_meld_v3_r4` |
| 用户显式 `checkpoint_preset` | 覆盖自动建议 |

### 12.3 为何仅 MELD 中文微调

| 数据集 | 中文微调？ | 理由 |
|--------|------------|------|
| **MELD** | **是** | 对话文本强依赖；P4 显示 T≈AVT；中文词表必须适配 |
| MOSEI | **否** | 英文 YouTube；换中文 BERT 无监督；保留 `sdavt_mosei_r4` 实验 |
| CREMA | **否** | P4 文本 collapse；应用靠 AV；保留 `sdavt_crema_r4` 实验 |

### 12.4 前端 optgroup 建议

```
├─ 推荐部署
│   ├─ [P0] sdavt_meld_zh_agent_v2（中文，F1=0.611）← 默认
│   └─ [P1] sdavt_meld_v3_r4（英文，F1=0.696）
├─ 中文对照
│   └─ [P2] sdavt_meld_zh_agent（v1，F1=0.601）
├─ 实验（单域）
│   ├─ [P3] sdavt_mosei_r4（F1=0.679）
│   └─ [P4] sdavt_crema_r4（C4_C3 Acc=0.605）
└─ 历史
    └─ [P5] ap2_m1（Acc≈0.61）
```

### 12.5 路径速查（相对 `PROJECT_ROOT`）

| Preset | Config | Checkpoint |
|--------|--------|------------|
| `sdavt_meld_zh_agent_v2` | `config/sdavt_v3_r4/p3_m3/meld/M3_M7_chinese_agent_v2.yaml` | `checkpoints_sdavt_v3_r4/SDAVT_R4_M3_M7_chinese_agent_v2/checkpoint_finetune_best_f1.pth` |
| `sdavt_meld_v3_r4` | `config/sdavt_v3_r4/p3_m3/meld/M3_M7_combo.yaml` | `checkpoints_sdavt_v3_r4/SDAVT_R4_M3_M7_combo/checkpoint_pretrain_best_f1.pth` |
| `sdavt_meld_zh_agent` | `config/sdavt_v3_r4/p3_m3/meld/M3_M7_chinese_agent.yaml` | `checkpoints_sdavt_v3_r4/SDAVT_R4_M3_M7_chinese_agent/checkpoint_finetune_best_f1.pth` |
| `sdavt_mosei_r4` | `config/sdavt_v3_r4/p2_fusion/mosei/F_O_ES_emotion_shift.yaml` | `checkpoints_sdavt_v3_r4/SDAVT_R4_F_O_ES_20260624_101647/checkpoint_pretrain_best_f1.pth` |
| `sdavt_crema_r4` | `config/sdavt_v3_r4/p3_c_plus/crema/C4_C3_c3_warmstart_acc.yaml` | `checkpoints_sdavt_v3_r4/SDAVT_R4_C4_C3_c3_warmstart_acc/checkpoint_pretrain_best_f1.pth` |

---

## 13. 全量 run Best/Last 指标表（磁盘复核）

> **复核时间：2026-07-16**。源：`project/logs_sdavt_v3_r4/*/metrics.csv` 中 `phase=val`。  
> 共 **59** 个含 metrics 的 run（队列正式结果 + close-out + 中文 v1/v2；不含已归档 failed C4_C1）。  
> **Best F1 / Best Acc** 可为不同 epoch；部署取 **best_f1**（CREMA close-out 取 best Acc 导向 ckpt）。  
> 中文 v2 的 Best 取自 CSV **全量段**（见 §10.3）；表中 n_val=23 含 smoke+full 拼接行。

| Run Dir | n_val | Best F1 @ep | Best Acc @ep | Last F1 | Last Acc | Last ep | CKPT_f1 |
|---------|------:|-------------|--------------|---------|----------|--------:|:-------:|
| `SDAVT_R4_C3_C1_baseline_20260625_200937` | 45 | 0.5336 @29 | 0.5417 @29 | 0.4956 | 0.5108 | 44 | ✓ |
| `SDAVT_R4_C3_C2_w2v_large_20260626_004150` | 47 | 0.5629 @31 | 0.5672 @31 | 0.5392 | 0.5511 | 46 | ✓ |
| `SDAVT_R4_C3_C3_focal_20260626_043125` | 50 | 0.5526 @41 | 0.5565 @41 | 0.5392 | 0.5390 | 49 | ✓ |
| `SDAVT_R4_C4_C2_c3_base_acc` | 60 | 0.3152 @48 | 0.3535 @51 | 0.3034 | 0.3441 | 59 | ✓ |
| `SDAVT_R4_C4_C3_c3_warmstart_acc` | 48 | 0.6057 @65 | 0.6048 @65 | 0.5574 | 0.5618 | 79 | ✓ |
| `SDAVT_R4_F_C_ES` | 41 | 0.5786 @28 | 0.5860 @28 | 0.4959 | 0.5040 | 40 | ✓ |
| `SDAVT_R4_F_C_LFA_20260624_070022` | 28 | 0.2978 @26 | 0.3065 @15 | 0.2906 | 0.3024 | 27 | ✓ |
| `SDAVT_R4_F_C_LFT_20260624_073903` | 28 | 0.3081 @15 | 0.3320 @15 | 0.2809 | 0.3065 | 27 | ✓ |
| `SDAVT_R4_F_C_STD_20260624_081739` | 27 | 0.2405 @14 | 0.2594 @4 | 0.1923 | 0.1989 | 26 | ✓ |
| `SDAVT_R4_F_C_TS_20260622_140751` | 28 | 0.2579 @15 | 0.2836 @27 | 0.2367 | 0.2836 | 27 | ✓ |
| `SDAVT_R4_F_C_TS_20260624_085905` | 42 | 0.2436 @29 | 0.3051 @27 | 0.2199 | 0.2728 | 41 | ✓ |
| `SDAVT_R4_F_M_ES` | 16 | 0.6109 @3 | 0.6245 @3 | 0.4522 | 0.4341 | 15 | ✓ |
| `SDAVT_R4_F_M_LFA_20260623_210443` | 12 | 0.4562 @3 | 0.4937 @1 | 0.4063 | 0.4558 | 11 | ✓ |
| `SDAVT_R4_F_M_LFT_20260623_225301` | 14 | 0.4207 @5 | 0.4486 @9 | 0.3978 | 0.4242 | 13 | ✓ |
| `SDAVT_R4_F_M_STD_20260622_140751` | 13 | 0.4595 @4 | 0.4991 @2 | 0.4260 | 0.4404 | 12 | ✓ |
| `SDAVT_R4_F_M_STD_20260624_005729` | 14 | 0.4447 @5 | 0.4973 @2 | 0.4239 | 0.4440 | 13 | ✓ |
| `SDAVT_R4_F_M_TS_20260622_163603` | 18 | 0.3826 @9 | 0.4368 @9 | 0.3000 | 0.3051 | 17 | ✓ |
| `SDAVT_R4_F_M_TS_20260624_030111` | 21 | 0.3682 @12 | 0.4314 @5 | 0.3413 | 0.3827 | 20 | ✓ |
| `SDAVT_R4_F_O_ES_20260624_101647` | 21 | 0.6792 @12 | 0.7269 @9 | 0.6746 | 0.7114 | 20 | ✓ |
| `SDAVT_R4_F_O_LFT_20260622_214330` | 9 | 0.5980 @7 | 0.7162 @0 | 0.5978 | 0.7162 | 8 | ✓ |
| `SDAVT_R4_F_O_LFT_20260624_110504` | 9 | 0.5980 @7 | 0.7162 @0 | 0.5978 | 0.7162 | 8 | ✓ |
| `SDAVT_R4_F_O_STD_20260622_222125` | 9 | 0.5988 @7 | 0.7162 @0 | 0.5987 | 0.7141 | 8 | ✓ |
| `SDAVT_R4_F_O_STD_20260624_115119` | 9 | 0.5988 @7 | 0.7162 @0 | 0.5987 | 0.7141 | 8 | ✓ |
| `SDAVT_R4_F_O_TS_20260622_232044` | 9 | 0.5978 @0 | 0.7162 @0 | 0.5962 | 0.7130 | 8 | ✓ |
| `SDAVT_R4_F_O_TS_20260624_123518` | 9 | 0.5978 @0 | 0.7162 @0 | 0.5962 | 0.7130 | 8 | ✓ |
| `SDAVT_R4_M3_M0_baseline_20260625_200937` | 19 | 0.6080 @3 | 0.6218 @3 | 0.4524 | 0.4495 | 18 | ✓ |
| `SDAVT_R4_M3_M1_roberta` | 24 | 0.6823 @17 | 0.6968 @22 | 0.6777 | 0.6922 | 23 | ✓ |
| `SDAVT_R4_M3_M2_w2v_large_20260625_233919` | 19 | 0.5572 @3 | 0.6020 @3 | 0.4176 | 0.4801 | 18 | ✓ |
| `SDAVT_R4_M3_M3_uniform_20260626_031222` | 19 | 0.6105 @3 | 0.6245 @3 | 0.4418 | 0.4477 | 18 | ✓ |
| `SDAVT_R4_M3_M4_focal_20260626_062916` | 19 | 0.6079 @3 | 0.6209 @3 | 0.4515 | 0.4540 | 18 | ✓ |
| `SDAVT_R4_M3_M5_context_20260626_073046` | 19 | 0.5725 @3 | 0.5912 @5 | 0.4069 | 0.4386 | 18 | ✓ |
| `SDAVT_R4_M3_M6_moddrop_20260626_113831` | 22 | 0.6079 @6 | 0.6245 @6 | 0.5194 | 0.5397 | 21 | ✓ |
| `SDAVT_R4_M3_M7_chinese_agent` | 14 | 0.6010 @9 | 0.6336 @13 | 0.5973 | 0.6336 | 13 | ✓ |
| `SDAVT_R4_M3_M7_chinese_agent_v2` | 23 | 0.6114 @5 | 0.6363 @5 | 0.6020 | 0.6336 | 10 | ✓ |
| `SDAVT_R4_M3_M7_combo` | 33 | 0.6957 @31 | 0.7121 @31 | 0.6880 | 0.7058 | 32 | ✓ |
| `SDAVT_R4_R4_A_C_A` | 43 | 0.1412 @30 | 0.1909 @33 | 0.1392 | 0.1841 | 42 | ✓ |
| `SDAVT_R4_R4_A_C_AT` | 29 | 0.1312 @16 | 0.1815 @11 | 0.1141 | 0.1680 | 28 | ✓ |
| `SDAVT_R4_R4_A_C_AV` | 45 | 0.3303 @32 | 0.3562 @41 | 0.3167 | 0.3427 | 44 | ✓ |
| `SDAVT_R4_R4_A_C_AVT` | 49 | 0.3263 @36 | 0.3575 @36 | 0.3146 | 0.3468 | 48 | ✓ |
| `SDAVT_R4_R4_A_C_T` | 21 | 0.0891 @8 | 0.1761 @8 | 0.0520 | 0.1747 | 20 | ✓ |
| `SDAVT_R4_R4_A_C_V` | 50 | 0.3538 @37 | 0.3804 @25 | 0.3519 | 0.3763 | 49 | ✓ |
| `SDAVT_R4_R4_A_C_VT` | 24 | 0.3159 @21 | 0.3481 @21 | 0.2827 | 0.3306 | 23 | ✓ |
| `SDAVT_R4_R4_A_M_A` | 49 | 0.4821 @33 | 0.5072 @12 | 0.4657 | 0.4847 | 48 | ✓ |
| `SDAVT_R4_R4_A_M_AT` | 11 | 0.6736 @5 | 0.6913 @5 | 0.6624 | 0.6697 | 10 | ✓ |
| `SDAVT_R4_R4_A_M_AV` | 37 | 0.4780 @21 | 0.5054 @5 | 0.4545 | 0.4829 | 36 | ✓ |
| `SDAVT_R4_R4_A_M_AVT` | 21 | 0.6818 @5 | 0.6958 @5 | 0.6605 | 0.6652 | 20 | ✓ |
| `SDAVT_R4_R4_A_M_T` | 22 | 0.6741 @6 | 0.6895 @6 | 0.6510 | 0.6588 | 21 | ✓ |
| `SDAVT_R4_R4_A_M_V` | 25 | 0.2690 @9 | 0.4233 @0 | 0.2518 | 0.4233 | 24 | ✓ |
| `SDAVT_R4_R4_A_M_VT` | 22 | 0.6738 @6 | 0.6895 @6 | 0.6569 | 0.6670 | 21 | ✓ |
| `SDAVT_R4_R4_A_O_A` | 22 | 0.6362 @13 | 0.7162 @0 | 0.6256 | 0.7146 | 21 | ✓ |
| `SDAVT_R4_R4_A_O_AT` | 15 | 0.6922 @6 | 0.7376 @12 | 0.6812 | 0.7189 | 14 | ✓ |
| `SDAVT_R4_R4_A_O_AV` | 11 | 0.6415 @2 | 0.7162 @1 | 0.6412 | 0.6750 | 10 | ✓ |
| `SDAVT_R4_R4_A_O_AVT` | 15 | 0.6982 @11 | 0.7338 @7 | 0.6898 | 0.7146 | 14 | ✓ |
| `SDAVT_R4_R4_A_O_T` | 24 | 0.7087 @15 | 0.7483 @22 | 0.7073 | 0.7402 | 23 | ✓ |
| `SDAVT_R4_R4_A_O_V` | 9 | 0.6274 @0 | 0.7162 @1 | 0.5978 | 0.7162 | 8 | ✓ |
| `SDAVT_R4_R4_A_O_VT` | 28 | 0.7050 @19 | 0.7483 @26 | 0.7002 | 0.7349 | 27 | ✓ |
| `SDAVT_R4_R4_B_C0_20260623_005627` | 32 | 0.5889 @19 | 0.5874 @19 | 0.5290 | 0.5242 | 31 | ✓ |
| `SDAVT_R4_R4_B_M1_20260623_005627` | 14 | 0.5680 @3 | 0.5966 @2 | 0.3932 | 0.4070 | 13 | ✓ |
| `SDAVT_R4_R4_B_O0_20260623_032138` | 21 | 0.6792 @12 | 0.7269 @9 | 0.6746 | 0.7114 | 20 | ✓ |

### 13.1 读表要点（效果速览）

| 观察 | 依据 | 含义 |
|------|------|------|
| MELD 英文冠军稳定 | M3_M7 Last≈0.688 vs Best 0.696 | 适合论文主表与英文 Agent |
| ES 早峰需 best ckpt | F_M_ES Last 0.45≪Best 0.61 | 禁止用末轮权重部署 |
| MOSEI 很稳 | F_O_ES / B_O0 Last≈Best | 实验 preset 可靠 |
| CREMA close-out 有回落 | C4_C3 Last Acc 0.56 vs Best 0.60 | 仍报 Best；未达 Tier-2 0.63 |
| 中文 v2 可用 | Best 0.6114，Last 0.602 | 默认部署；与英文分表 |

---

## 14. Checkpoint 与日志清单

- 权重根目录：`checkpoints_sdavt_v3_r4/`（与上表 Run Dir 同名子目录；多数含 `checkpoint_pretrain_best_f1.pth`，中文为 `checkpoint_finetune_best_f1.pth`）。  
- 失败归档：`logs_sdavt_v3_r4_archived/SDAVT_R4_C4_C1_combo_acc`（failed_nan）。  
- 早期 P0/P2 重试 run（同 job 多时间戳）保留在日志桶内，论文主表以 **无时间戳正式名** 或 **P2 正式批次**（如 `F_M_ES`、`F_O_ES_20260624_*`）为准。  
- 中文状态日志：`outputs_sdavt_v3_r4/status/m3m7_zh_finetune.log`、`m3m7_zh_v2_finetune.log`（smoke）、`m3m7_zh_v2_full_finetune.log`（全量，`EXIT_CODE=0`）。  
- TensorBoard：各 run 目录下 `events.out.tfevents.*`，统一 logdir 见 §4（端口 **6008**）。

**Agent 默认权重：**

```text
checkpoints_sdavt_v3_r4/SDAVT_R4_M3_M7_chinese_agent_v2/checkpoint_finetune_best_f1.pth
```

---

## 15. 修订记录

| 日期 | 版本 | 摘要 |
|------|------|------|
| 2026-07-09 | v1.0 | 自动生成：55 jobs + close-out + 分阶段表 |
| 2026-07-16 | v1.1 | 纳入中文微调 v1/v2；深化消融；新增 Agent Preset 矩阵 |
| 2026-07-16 | **v1.2** | **59 组 metrics 全量 Best/Last 复核**（§13）；补中文逐 epoch / smoke+full 说明；稳定性 §1.1；Preset 表对齐 `config.py`；修正 C4_C1 归档路径 |

---

*刷新队列简表：`python scripts/build_sdavt_r4_report.py` → `SDAVT_V3_R4_EXPERIMENT_RESULTS.md`*  
*本文件为 R4+微调权威长文；Preset 代码：`emotion-agent/backend/app/core/config.py`*
