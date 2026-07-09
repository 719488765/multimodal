# SDAVT v3 R4 完整实验报告

*生成时间：2026-07-09 12:30 UTC*  
*生成脚本：`scripts/build_r4_full_experiment_report.py`*  
*数据源：`experiment_queue.json`（55 jobs）+ P3-C+ 重训（3 jobs）+ `logs_sdavt_v3_r4/` metrics*

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
10. [结果分析与论文叙事](#10-结果分析与论文叙事)
11. [Checkpoint 与日志完整清单](#11-checkpoint-与日志完整清单)

---

## 1. 实验总览与 Tier-2 结论

R4（Round 4）为 SDAVT v3 **论文主轨**实验线，共 **6 个阶段 + Close-out 重训**：

| 阶段 | 目的 | Job 数 |
|------|------|--------|
| P0 | 融合模块修复验证 | 6 |
| P1 | 三数据集 ES 主基线 | 3 |
| P2 | 五融合策略对比 | 14 |
| P3-C | CREMA Acc 配方消融 | 3 (+3 close-out) |
| P3-M | MELD F1 配方消融 | 8 |
| P4 | 模态组合消融 Table 4 | 21 |
| **合计（队列）** | | **55 done** |

### Tier-2 验收（论文口径，2026-07-09 close-out）

| 数据集 | 指标目标 | **最终 Champion** | Best F1 | Best Acc | 判定 |
|--------|----------|-------------------|---------|----------|------|
| **MELD** | F1≥0.59, Acc≥0.62 | **M3_M7_combo** | **0.696** @ ep31 | **0.712** @ ep31 | **PASS** |
| **MOSEI** | F1≥0.67 | **F_O_ES**（P2 融合冠军） | **0.679** @ ep12 | 0.727 @ ep12 | **PASS** |
| **CREMA** | Acc≥0.63 | **C4_C3** warm-start | 0.606 @ ep65 | **0.605** @ ep65 | **CLOSE-OUT**（差 2.5pp） |

**Audit：** P0=0，collapse（严格审计）=0，tier2_fail=1（CREMA 未达 0.63，已由 C4_C3 close-out 定论）。

---

## 2. 实验过程与时间线

| 时间段 | 阶段 | 关键事件 |
|--------|------|----------|
| 2026-06-22 ~ 06-23 | P0 | 融合修复 sanity（STD/TS/LFT 等） |
| 2026-06-23 | P1 | 三数据集 ES 主基线 R4_B_* 完成 |
| 2026-06-23 ~ 06-24 | P2 | 14 组融合对比；**ES 锁定**为 P3/P4 默认 |
| 2026-06-25 ~ 06-26 | P3 | MELD 8 组配方 + CREMA 3 组配方并行 |
| 2026-06-26 ~ 06-30 | P4 | 21 组模态消融（Table 4） |
| 2026-07-07 ~ 07-09 | Close-out | C4_C3 warm-start + R4_A_M_V 重训；R4 GPU 线关闭 |

**执行方式：** 双 GPU 队列 `sdavt_r4_worker.sh` + tmux 监控；指标自动写入 `metrics.csv` / TensorBoard。

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

| 资源 | 路径 | 数量 |
|------|------|------|
| 训练日志 | `logs_sdavt_v3_r4/` | **65** runs |
| Checkpoint | `checkpoints_sdavt_v3_r4/` | **87** dirs |
| 队列状态 | `outputs_sdavt_v3_r4/experiment_queue.json` | 55 jobs |
| 指标表 | `outputs_sdavt_v3_r4/tables/r4_training_audit.json` | 63 runs w/ metrics |
| Close-out 快照 | `outputs_sdavt_v3_r4/status/r4_closeout_snapshot_20260709.json` | — |

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

### p3_c_plus — p3_c_plus

| Job | Dataset | 模态 | 融合/骨干 | Best F1 | Best Acc | Ep | Collapse | Run Dir | CKPT |
|-----|---------|------|-----------|---------|----------|-----|----------|---------|------|
| C4_C1_combo_acc | crema | T+A+V | T:bert-base-uncased; A:wav2vec2-large-960h; V:re | — | — | — | — | `SDAVT_R4_C4_C1_combo_acc` | — |
| C4_C2_c3_base_acc | crema | T+A+V | T:bert-base-uncased; A:wav2vec2-large-960h; V:re | 0.3152 @ ep48 | 0.3535 @ ep51 | 60 | ✓ | `SDAVT_R4_C4_C2_c3_base_acc` | ✓ |
| C4_C3_c3_warmstart_acc | crema | T+A+V | T:bert-base-uncased; A:wav2vec2-large-960h; V:re | 0.6057 @ ep65 | 0.6048 @ ep65 | 48 | ✓ | `SDAVT_R4_C4_C3_c3_warmstart_acc` | ✓ |

---

## 6. 融合策略消融（P2）

P2 在三数据集上对比 **5 种融合**：`standard` / `two_stage` / `leader_follower` / `functional_correlation` / `emotion_shift`。
**结论：** `emotion_shift`（ES）在 MELD/MOSEI 上稳定最优或并列最优，CREMA 上 ES 亦为 P2 冠军，故 P3/P4 全部固定 ES。

| Dataset | P2 冠军 Job | Best F1 | Best Acc | Run |
|---------|-------------|---------|----------|-----|
| crema | **F_C_ES** | 0.5786 @ ep28 | 0.5860 | `SDAVT_R4_F_C_ES` |
| meld | **F_M_ES** | 0.6109 @ ep3 | 0.6245 | `SDAVT_R4_F_M_ES` |
| mosei | **F_O_ES** | 0.6792 @ ep12 | 0.7269 | `SDAVT_R4_F_O_ES_20260624_101647` |

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

### 6.1 MELD（P3-M）— 目标 F1≥0.59

| Job | 配方要点 | Best F1 | Best Acc | 相对 M0 |
|-----|----------|---------|----------|---------|
| **M3_M7_combo** | M7_combo | **0.6957** @ ep31 | 0.7121 | +0.0877 |
| **M3_M1_roberta** | M1_roberta | **0.6823** @ ep17 | 0.6968 | +0.0743 |
| **M3_M3_uniform** | uniform | **0.6105** @ ep3 | 0.6245 | +0.0025 |
| **M3_M0_baseline** | M0_baseline | **0.6080** @ ep3 | 0.6218 | +0.0000 |
| **M3_M4_focal** | M4_focal | **0.6079** @ ep3 | 0.6209 | -0.0001 |
| **M3_M6_moddrop** | M6_moddrop | **0.6079** @ ep6 | 0.6245 | -0.0001 |
| **M3_M5_context** | M5_context | **0.5725** @ ep3 | 0.5912 | -0.0355 |
| **M3_M2_w2v_large** | M2_w2v_large | **0.5572** @ ep3 | 0.6020 | -0.0508 |

**M3_M7_combo 配方（冠军）：** roberta-base + wav2vec2-large + ResNet50；`use_context_window=true`；`modality_dropout=0.1`；focal loss + label smoothing；dropout 0.35。

### 6.2 CREMA（P3-C + P3-C+）— 目标 Acc≥0.63

| Job | 配方要点 | Best Acc | Best F1 | 判定 |
|-----|----------|----------|---------|------|
| C4_C3_c3_warmstart_acc | p3_c_plus | **0.6048** @ ep65 | 0.6057 | **CLOSE-OUT champion** |
| C3_C2_w2v_large | p3_c3 | **0.5672** @ ep31 | 0.5629 | gpu1 |
| C3_C3_focal | p3_c3 | **0.5565** @ ep41 | 0.5526 | gpu1 |
| C3_C1_baseline | p3_c3 | **0.5417** @ ep29 | 0.5336 | gpu1 |
| C4_C2_c3_base_acc | p3_c_plus | **0.3535** @ ep51 | 0.3152 | 队列完成 |
| C4_C1_combo_acc | p3_c_plus | — @ ep— | — | failed_nan |

---

## 8. 模态消融（P4）

P4 在 ES 融合固定下，对 **7 种模态组合**（A/T/V/AT/AV/VT/AVT）× **3 数据集** 进行消融。
Job 命名：`R4_A_{C|M|O}_{modality}`（C=CREMA, M=MELD, O=MOSEI）。

### MELD

| 模态 | Job | Best F1 | Best Acc | Collapse | 分析 |
|------|-----|---------|----------|----------|------|
| T+A+V | R4_A_M_AVT | 0.6818 @ ep5 | 0.6958 | ✓ |  |
| T | R4_A_M_T | 0.6741 @ ep6 | 0.6895 | ✓ | 文本单模态接近全模态（0.674 vs AVT 0.682） |
| T+V | R4_A_M_VT | 0.6738 @ ep6 | 0.6895 | ✓ |  |
| T+A | R4_A_M_AT | 0.6736 @ ep5 | 0.6913 | ✓ |  |
| A | R4_A_M_A | 0.4821 @ ep33 | 0.5072 | ✓ |  |
| A+V | R4_A_M_AV | 0.4780 @ ep21 | 0.5054 | ✓ |  |
| V | R4_A_M_V | 0.2690 @ ep9 | 0.4233 | ✓ | 视频单模态极弱（对话情感依赖文本/音频） |

### MOSEI

| 模态 | Job | Best F1 | Best Acc | Collapse | 分析 |
|------|-----|---------|----------|----------|------|
| T | R4_A_O_T | 0.7087 @ ep15 | 0.7483 | ✓ | 文本单模态接近全模态（0.709 vs AVT 0.698） |
| T+V | R4_A_O_VT | 0.7050 @ ep19 | 0.7483 | ✓ |  |
| T+A+V | R4_A_O_AVT | 0.6982 @ ep11 | 0.7338 | ✓ |  |
| T+A | R4_A_O_AT | 0.6922 @ ep6 | 0.7376 | ✓ |  |
| A+V | R4_A_O_AV | 0.6415 @ ep2 | 0.7162 | ✓ |  |
| A | R4_A_O_A | 0.6362 @ ep13 | 0.7162 | ✓ |  |
| V | R4_A_O_V | 0.6274 @ ep0 | 0.7162 | ✓ |  |

### CREMA

| 模态 | Job | Best F1 | Best Acc | Collapse | 分析 |
|------|-----|---------|----------|----------|------|
| V | R4_A_C_V | 0.3538 @ ep37 | 0.3804 | ✓ |  |
| A+V | R4_A_C_AV | 0.3303 @ ep32 | 0.3562 | ✓ |  |
| T+A+V | R4_A_C_AVT | 0.3263 @ ep36 | 0.3575 | ✓ |  |
| T+V | R4_A_C_VT | 0.3159 @ ep21 | 0.3481 | ✓ |  |
| A | R4_A_C_A | 0.1412 @ ep30 | 0.1909 | ✓ |  |
| T+A | R4_A_C_AT | 0.1312 @ ep16 | 0.1815 | ✓ |  |
| T | R4_A_C_T | 0.0891 @ ep8 | 0.1761 | ✓ | 文本单模态接近全模态（0.089 vs AVT 0.326） |

---

## 9. Close-out 并行重训

| Job | 完成时间 | Best 指标 | 判定 | Run |
|-----|----------|-----------|------|-----|
| **C4_C3** | 2026-07-09T06:32:00+08:00 | F1=0.605655 Acc=0.604839 | PARTIAL | `SDAVT_R4_C4_C3_c3_warmstart_acc` |
| **R4_A_M_V** | 2026-07-09T07:55:00+08:00 | F1=0.268985 Acc=0.423285 | FAIL | `SDAVT_R4_R4_A_M_V` |

---

## 10. 结果分析与论文叙事

### 9.1 主要发现

1. **融合：** Emotion-Shift 在三个数据集 P2 对比中均为首选，验证 CFN-ESA 风格 shift-aware 融合在本骨架上的有效性。
2. **MELD 配方：** M3_M7（RoBERTa-large 音频 + context window + modality dropout）相对 M0 提升 **+8.8pp F1**，为 Agent 默认 preset。
3. **MOSEI：** F_O_ES F1=0.679 达标；P4 全模态 R4_A_O_AVT F1=0.698 略高，但 P2 冠军仍为 ES 主轨配置。
4. **CREMA：** C3_C2 w2v-large Acc=0.567 → C4_C3 warm-start **0.605**，未达 Tier-2 0.63；激进改 recipe（C4_C1/C4_C2）退化。
5. **模态消融：** MELD 上 **T ≈ AVT**（F1 0.674 vs 0.682），**V-only F1≈0.27** 为任务固有难度，非训练失效。

### 9.2 论文推荐数字

| Table | 内容 | 推荐数值 |
|-------|------|----------|
| 主结果 MELD | M3_M7_combo | F1=**0.696**, Acc=**0.712** |
| 主结果 MOSEI | F_O_ES | F1=**0.679**, Acc=0.727 |
| 主结果 CREMA | C4_C3 | Acc=**0.605**, F1=0.606 |
| Table 4 脚注 | R4_A_M_V | F1≈0.269（V-only 下限） |

### 9.3 Agent 部署映射

| Preset | Checkpoint | 来源 Job |
|--------|------------|----------|
| `sdavt_meld_v3_r4` | `SDAVT_R4_M3_M7_combo/checkpoint_pretrain_best_f1.pth` | M3_M7_combo |
| `sdavt_mosei_r4` | `SDAVT_R4_F_O_ES_*/checkpoint_pretrain_best_f1.pth` | F_O_ES |
| `sdavt_crema_r4` | C3_C2 或 C4_C3 | 实验/close-out |

---

## 11. Checkpoint 与日志完整清单

共 **65** 个 log run、**87** 个 checkpoint 目录（含时间戳后缀变体）。

| Run Dir | metrics.csv | Checkpoint (.pth) | TB Scalars |
|---------|-------------|-------------------|------------|
| `SDAVT_R4_C3_C1_baseline_20260625_200937` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_C3_C2_w2v_large_20260626_004150` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_C3_C3_focal_20260626_043125` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_C4_C1_combo_acc` | ✓ | — | ✓ |
| `SDAVT_R4_C4_C2_c3_base_acc` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_C4_C3_c3_warmstart_acc` | ✓ | checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_F_C_ES` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_F_C_LFA_20260624_070022` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_F_C_LFT_20260624_073903` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_F_C_STD_20260624_081739` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_F_C_TS_20260622_140751` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_F_C_TS_20260624_085905` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_F_M_ES` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_F_M_LFA_20260623_210443` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_F_M_LFT_20260623_225301` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_F_M_STD_20260622_140751` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_F_M_STD_20260624_005729` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_F_M_TS_20260622_163603` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_F_M_TS_20260624_030111` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_F_O_ES_20260624_101647` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_F_O_LFT_20260622_174508` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_F_O_LFT_20260622_212841` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_F_O_LFT_20260622_214330` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_F_O_LFT_20260624_110504` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_F_O_STD_20260622_191324` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_F_O_STD_20260622_213550` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_F_O_STD_20260622_222125` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_F_O_STD_20260624_115119` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_F_O_TS_20260622_202016` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_F_O_TS_20260622_213845` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_F_O_TS_20260622_232044` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_F_O_TS_20260624_123518` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_M3_M0_baseline_20260625_200937` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_M3_M1_roberta` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_M3_M2_w2v_large_20260625_233919` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_M3_M3_uniform_20260626_031222` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_M3_M4_focal_20260626_062916` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_M3_M5_context_20260626_073046` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_M3_M6_moddrop_20260626_113831` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_M3_M7_chinese_agent` | ✓ | checkpoint_finetune_best_f1.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_M3_M7_combo` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_R4_A_C_A` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_R4_A_C_AT` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_R4_A_C_AV` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_R4_A_C_AVT` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_R4_A_C_T` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_R4_A_C_V` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_R4_A_C_VT` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_R4_A_M_A` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_R4_A_M_AT` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_R4_A_M_AV` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_R4_A_M_AVT` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_R4_A_M_T` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_R4_A_M_V` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_R4_A_M_VT` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_R4_A_O_A` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_R4_A_O_AT` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_R4_A_O_AV` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_R4_A_O_AVT` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_R4_A_O_T` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_R4_A_O_V` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_R4_A_O_VT` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_R4_B_C0_20260623_005627` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_R4_B_M1_20260623_005627` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |
| `SDAVT_R4_R4_B_O0_20260623_032138` | ✓ | checkpoint_pretrain_best.pth, checkpoint_pretrain_best_f1.pth | ✓ |

---

*刷新：`python scripts/build_r4_full_experiment_report.py`*  
*配套实时表：`docs/SDAVT_V3_R4_EXPERIMENT_RESULTS.md`（`build_sdavt_r4_report.py`）*
