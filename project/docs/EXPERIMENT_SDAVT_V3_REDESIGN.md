# SDAVT v3 实验重设计方案（Single-Domain AVT）

**版本**：v3.11（2026-06-30）  
**状态**：**R4 论文主轨进行中** — P0/P1/P2/P2.5/P3 **完成**；P4 **MELD 7/7 + MOSEI 4/7 完成**，**CREMA 7 jobs 重跑中**（已修复 log 槽位污染）  
**前置条件**：CREMA / MOSEI / MELD 模态已补全；融合架构修复已验收（§17.3）

> **阅读路线**：设计原则 §1–§6 → R4 工作日志 **§17** → Tier-2 与 P3+ **§18**  
> **实时指标**：[`SDAVT_V3_R4_EXPERIMENT_RESULTS.md`](SDAVT_V3_R4_EXPERIMENT_RESULTS.md)（`bash scripts/start_sdavt_r4.sh report`）

---

## 1. 背景与动机

### 1.1 为何放弃三混合训练

| 问题 | 说明 |
|------|------|
| 标签空间不兼容 | CREMA 6 类 vs MELD/MOSEI 7 类；unified 映射引入噪声 |
| 域偏移 | 三域混合 val Acc 上限约 57–61%，无法发挥单域数据质量 |
| 历史缺陷 | 混合 batch 中 MOSEI video 曾置零；ClassBalanced 与 val loss 脱钩；无早停导致 Last 崩溃 |

### 1.2 本轮策略（SDAVT v3）

- **单域 AVT 三模态** × CREMA / MELD / MOSEI，**原生或统一标签**（按数据集策略）
- 数据已就绪：CREMA ASR 7442/7442、MOSEI COVAREP+SDK 文本 22860/22860
- **R4 论文主轨**（2026-06-22 起）：融合架构修复后全量重跑，与 AP/S1–S3 路径隔离

### 1.3 MELD plan 对齐（v3.1 冲突消解）

| 维度 | 统一决议 |
|------|----------|
| 日志路径 | 一律 `logs_sdavt_v3/`（S1–S3）→ **`logs_sdavt_v3_r4/`（R4 论文）** |
| MELD 主 run | S1-M0（AP1 对齐）+ S1-M1（v2 配方）→ R4 **M3_M7_combo** 冠军 |
| 指标口径 | 先 unified（与 AP1 0.58 可比）→ P3 native / Tier-2 |
| 早停 | `train.py` 已实现；R4 各 yaml 显式配置 |

---

## 2. 历史实验诊断摘要

| 轮次 | 目录 | MELD Best | CREMA Best | MOSEI Best | 结论 |
|------|------|-----------|------------|------------|------|
| AP1 | `logs_accuracy_seq/` | Acc 0.580 | ~0.34 | Acc 0.716 | 混合轨上限 |
| S1 | `logs_sdavt_v3/` | F1 0.427 | F1 0.322 | F1 0.088 | 配方/融合问题 |
| S2-M1 | `logs_sdavt_v3/` | **F1 0.605** | 失败 | 失败 | MELD 可达标 |
| S3 | `logs_sdavt_v3/` | 沿用 S2 | 修复中 | NaN 修复 | 为 R4 P0 前置 |
| **R4** | `logs_sdavt_v3_r4/` | **F1 0.696** | Acc 0.567 | F1 0.679 | **论文主数据** |

---

## 3. 路径隔离与目录规范

| 维度 | 历史（AP） | SDAVT v3（S1–S3） | **R4 论文主轨（当前）** |
|------|------------|-------------------|-------------------------|
| 配置 | `config/rerun/accuracy_plan/` | `config/sdavt_v3/` | **`config/sdavt_v3_r4/`** |
| 日志 | `logs_accuracy_seq/` | `logs_sdavt_v3/` | **`logs_sdavt_v3_r4/`** |
| Checkpoint | `checkpoints_accuracy_seq/` | `checkpoints_sdavt_v3/` | **`checkpoints_sdavt_v3_r4/`** |
| 汇总 | `outputs_accuracy_seq/` | `outputs_sdavt_v3/` | **`outputs_sdavt_v3_r4/`** |
| TensorBoard | :6006 系列 | `tensorboard_sdavt_v3.sh` (:6007) | **`tensorboard_sdavt_r4.sh` (:6008)** |

**禁止**：将 R4 run 写入 `logs_sdavt_v3/` 或与 AP/S1–S3 混排名。

### 3.1 R4 阶段流水线

```
P0 fix (6) → P1 baseline (3) → P2 fusion (14) → P2.5 ES anti-overfit 重训
  → P3-M (8) + P3-C (3) → P4 modal (21: MELD7 + CREMA7 + MOSEI7)
  → P5 Agent preset + 论文表
```

### 3.2 指标目标

| 数据集 | P1 目标 | Tier-2（P3+） | **R4 当前最优** |
|--------|---------|---------------|-----------------|
| MELD | Acc≥0.60, F1≥0.58 | **F1≥0.59, Acc≥0.62** | **F1=0.696, Acc=0.712**（M3_M7_combo）✓ |
| CREMA | Acc≥0.50, F1≥0.45 | **Acc≥0.63** | Acc=**0.567**（C3_C2_w2v_large）△ |
| MOSEI | Acc≥0.70, F1≥0.62 | 维持 P2 | F1=**0.679**, Acc=**0.727** ✓ |

---

## 4. 分阶段实验设计（S1–S3 探索轨）

| 阶段 | 内容 | 状态 |
|------|------|------|
| **P0** | 原生标签、`label_mapping.py`、早停模板、数据审计 | ✓ 完成 |
| **P1** | 三单域 AVT 基线（S1-M0/M1, S1-C0, S1-O0） | ✓ 完成 |
| **P2** | S2 重训优化（见 `SDAVT_V3_S2_OPTIMIZATION.md`） | ✓ MELD 达标 |
| **P3** | S3 修复（见 `SDAVT_V3_S3_FIX.md`） | ✓ 完成 |

配置模板：`config/sdavt_v3/_template_sdavt_single_domain.yaml`  
校验脚本：`scripts/validate_sdavt_v3_prep.py`

---

## 5. 启动命令速查

```bash
cd project

# R4 主轨
bash scripts/start_sdavt_r4.sh status
bash scripts/start_sdavt_r4.sh report
bash scripts/start_sdavt_r4.sh run p4_crema_mosei

# TensorBoard
bash scripts/tensorboard_sdavt_r4.sh   # :6008

# 监控
bash scripts/tail_r4_training.sh status
python3 scripts/monitor_sdavt_r4.py --once
```

---

## 6. 论文表格结构

| 表 | 内容 | R4 Phase |
|----|------|----------|
| Table 1 | 三单域 ES 主基线 | p1_baseline |
| Table 2 | 融合策略消融（5×MELD + 5×CREMA + 4×MOSEI） | p2_fusion |
| Table 3 | MELD Tier-2 优化矩阵 | p3_m3 |
| Table 3b | CREMA Tier-2 补强 | p3_c3 |
| Table 4 | 模态消融（三数据集 × 7 组合） | p4_modal |

机器可读汇总：`outputs_sdavt_v3_r4/tables/r4_*.csv`

---

## 17. R4 论文主轨：资产目录、代码优化与工作日志

> **本章为 R4 轮次专属工作记录**（2026-06-22 起）。  
> **实时指标**以 [`SDAVT_V3_R4_EXPERIMENT_RESULTS.md`](SDAVT_V3_R4_EXPERIMENT_RESULTS.md) 为准。

### 17.1 本轮实验资产目录总览

#### 17.1.1 三层路径隔离

| 层级 | 用途 | 配置 | 日志 | Checkpoint | 汇总 |
|------|------|------|------|------------|------|
| **历史** | AP0–AP4 | `config/rerun/accuracy_plan/` | `logs_accuracy_seq/` | `checkpoints_accuracy_seq/` | `outputs_accuracy_seq/` |
| **SDAVT v3** | S1–S3、fix_v* | `config/sdavt_v3/` | `logs_sdavt_v3/` | `checkpoints_sdavt_v3/` | `outputs_sdavt_v3/` |
| **R4（论文）** | **当前主轨** | `config/sdavt_v3_r4/` | `logs_sdavt_v3_r4/` | `checkpoints_sdavt_v3_r4/` | `outputs_sdavt_v3_r4/` |

#### 17.1.2 运维命令

| 操作 | 命令 |
|------|------|
| 查看队列 | `bash scripts/start_sdavt_r4.sh status` |
| 刷新论文记录 | `bash scripts/start_sdavt_r4.sh report` |
| 启动 worker | `bash scripts/start_sdavt_r4.sh run <phase>` |
| P3 Tier-2 验收 | `python3 scripts/accept_sdavt_r4_p3_tier2.py --eval --refresh-report` |

#### 17.1.3 tmux 会话

| 会话 | 用途 |
|------|------|
| `sdavt_r4_worker_gpu0/1` | 双卡 R4 worker |
| `sdavt_r4_tb` | TensorBoard :6008 |

### 17.2 R4 分阶段结果（截至 2026-06-30）

#### 17.2.1 阶段进度

| Phase | 任务数 | 状态 | 说明 |
|-------|--------|------|------|
| **P0 fix** | 6 | ✓ 完成 | 融合修复验证；MOSEI 由 ln(7) 塌缩恢复 |
| **P1 baseline** | 3 | ✓ 完成 | R4_B_M1/C0/O0 |
| **P2 fusion** | 14 | ✓ 完成 | 三域冠军均为 emotion_shift |
| **P2.5** | 2 | ✓ 完成 | F_M_ES / F_C_ES anti-overfit 重训 |
| **P3-M** | 8 | ✓ 完成 | MELD Tier-2 **PASS** |
| **P3-C** | 3 | ✓ 完成 | CREMA Tier-2 **FAIL**（Acc 0.567） |
| **P4 MELD** | 7 | ✓ 完成 | 模态消融 |
| **P4 MOSEI** | 7 | 4 done + 3 pending | 进行中 |
| **P4 CREMA** | 7 | **重跑中** | 2026-06-30 修复 log 槽位 bug 后重置 |

#### 17.2.2 P1 主基线

| Job | Best F1 | Best Acc | 判定 |
|-----|---------|----------|------|
| R4_B_M1 | 0.568 | 0.597 | △ |
| R4_B_C0 | 0.589 | 0.587 | ✓ |
| R4_B_O0 | 0.679 | 0.727 | ✓ |

#### 17.2.3 P2 融合冠军

| 数据集 | Job | Best F1@ep | Best Acc@ep |
|--------|-----|------------|-------------|
| MELD | F_M_ES | 0.611 @ 3 | 0.625 @ 3 |
| CREMA | F_C_ES | 0.541 @ 34 | 0.547 @ 41 |
| MOSEI | F_O_ES | 0.679 @ 12 | 0.727 @ 9 |

#### 17.2.4 P3 冠军

| 数据集 | 冠军 Job | Best F1@ep | Best Acc@ep | Tier-2 |
|--------|----------|------------|-------------|--------|
| **MELD** | **M3_M7_combo** | **0.696 @ 31** | **0.712 @ 31** | **PASS** |
| MELD（次优） | M3_M1_roberta | 0.682 @ 17 | 0.697 @ 17 | PASS |
| **CREMA** | C3_C2_w2v_large | 0.563 @ 31 | 0.567 @ 31 | FAIL |

**MELD P3 关键发现**：
- P2.5 anti-overfit 将 MELD 拉过 Tier-2；P3 **M7 combo**（RoBERTa + w2v-large + focal + moddrop）进一步 +8.5 pt F1。
- 文本单模态 R4_A_M_T F1=0.626，接近 AVT，说明文本主导 MELD。

#### 17.2.5 P4 MELD 模态消融

| 模态 | Best F1 | Best Acc |
|------|---------|----------|
| T | **0.626** | **0.644** |
| AT | 0.623 | 0.640 |
| VT | 0.612 | 0.625 |
| AVT | 0.608 | 0.622 |
| A | 0.466 | 0.509 |
| AV | 0.434 | 0.487 |
| V | 0.384 | 0.438 |

### 17.3 代码优化清单

| 文件 | 改动摘要 |
|------|----------|
| `models/leader_follower_attention.py` | 修正 Q/K/V attention；数值稳定 |
| `models/multimodal_model.py` | non-ES fusion 统一走 `emotion_classifier` |
| `models/fusion_utils.py` | MOSEI npy 时序对齐 |
| `data/dataset.py` | `AutoTokenizer` 支持 RoBERTa（P3 修复） |
| `scripts/train.py` | ResNet 永久冻结；modality dropout；best_f1 ckpt 恢复 |
| `scripts/sdavt_r4_worker.sh` | phase+id 队列匹配 |
| `scripts/build_sdavt_r4_report.py` | 自动生成论文指标表 |

#### 17.3.5 MELD ES anti-overfit 配方

| 超参 | 旧值 | 新值 |
|------|------|------|
| freeze_backbone_epochs | 2 | **5** |
| backbone_lr_multiplier | 0.1 | **0.05** |
| dropout | 0.1 | **0.25** |
| weight_decay | 1e-5 | **1e-4** |
| label_smoothing | 0.0 | **0.05** |
| patience | 5 | **12** |

#### 17.3.6 P3 anti-overfit 重训（2026-06-29）

| 超参 | 值 |
|------|-----|
| dropout | 0.35 |
| weight_decay | 0.01 |
| freeze_backbone_epochs | 8 |
| backbone_lr_multiplier | 0.01 |
| ResNet | **永久冻结** |

### 17.4 工作日志时间线

| 日期 | 事项 |
|------|------|
| 06-16 | S2/S3 探索；SDAVT v3 前期 build 完成 |
| 06-22 | R4 启动；P0 MOSEI 塌缩诊断与融合架构修复 |
| 06-23 | P0/P1 完成；P2 14 jobs 启动 |
| 06-24 | P2 完成；F_M_ES 过拟合 → anti-overfit 重训 |
| 06-25 | P2.5 验收 PASS；P3 双卡启动 |
| 06-26 | P3-M/C 大部分完成；P4 MELD 启动 |
| 06-28 | P3 tokenizer 修复；M3_M1/M7 重跑启动 |
| 06-29 | P3 anti-overfit 重训；**M3_M7_combo 新冠军 F1=0.696** |
| 06-30 | P4 CREMA/MOSEI 扩展；**发现 CREMA P4 log_run_dir 污染 F_C_ES** → 已修复并重置队列 |
| 06-30 19:03 | **本文档意外被清空**（未入 git）；本节起从 agent transcript + 实验产物恢复 |

#### 17.4.1 已知问题与处置

| 问题 | 处置 | 状态 |
|------|------|------|
| MOSEI P0 ln(7) 塌缩 | 融合架构 + recipe 修复 | ✓ 已解决 |
| MELD P2 过拟合 | anti-overfit 配方 | ✓ 已解决 |
| P3 RoBERTa tokenizer | AutoTokenizer 修复 + 重跑 | ✓ 已解决 |
| **CREMA P4 共用 `log_run_dir: SDAVT_R4_F_C_ES`** | 删除错误配置；重置 7 jobs；归档污染 logs/ckpt | **2026-06-30 已修复** |
| F_C_ES anti-overfit ckpt 被覆盖 | 暂恢复 pre-antiof 备份；**待 P4 后重训 F_C_ES** | 待办 |

---

## 18. 参考文献差距与 P3+ Tier-2 重设计

> 完整对照：[`SDAVT_V3_REFERENCE_BENCHMARK.md`](SDAVT_V3_REFERENCE_BENCHMARK.md)（若存在）

### 18.1 Tier-2 验收（2026-06-30）

| 数据集 | 目标 | **当前最优** | 判定 |
|--------|------|-------------|------|
| MELD | F1≥0.59, Acc≥0.62 | **F1=0.696, Acc=0.712**（M3_M7_combo） | **PASS** |
| CREMA | Acc≥0.63 | Acc=0.567（C3_C2_w2v_large） | **FAIL** |
| MOSEI | F1≥0.67 | F1=0.679（F_O_ES） | PASS |

门禁文件：
- P2.5：`outputs_sdavt_v3_r4/status/p2_es_retrain_passed`
- P3 冠军：`outputs_sdavt_v3_r4/status/p3_m3_winner_meld.json`（**M3_M7_combo**）
- P3 验收：`outputs_sdavt_v3_r4/status/p3_tier2_acceptance_latest.md`

### 18.2 P5 Agent preset

**当前应使用 M3_M7_combo**（非旧版 M3_M3_uniform）：

```yaml
# config_agent_deploy.yaml → sdavt_meld_v3_r4
train_config: config/sdavt_v3_r4/p3_m3/meld/M3_M7_combo.yaml
checkpoint: checkpoints_sdavt_v3_r4/SDAVT_R4_M3_M7_combo/checkpoint_pretrain_best_f1.pth
```

```bash
bash scripts/apply_deploy_preset.sh sdavt_meld_v3_r4
```

### 18.3 下一步（2026-06-30）

1. **P4 CREMA 7 jobs 重跑**（配置已修复，worker 已重启）
2. **P4 MOSEI 剩余 3 jobs**（R4_A_O_T / V / VT）自动接续
3. **F_C_ES anti-overfit 重训**（P4 完成后，`bash scripts/retrain_r4_crema_p2_es.sh`）恢复 P2 冠军 ckpt
4. **刷新论文记录**：`bash scripts/start_sdavt_r4.sh report`
5. **P5**：更新 Agent preset 为 M3_M7_combo；撰写 Table 4 模态消融分析

---

## 16. 修订记录

| 日期 | 版本 | 变更 |
|------|------|------|
| 2026-06-14 | v3.2 | P0 build 完成；S1 配置就绪 |
| 2026-06-16 | v3.4 | S2/S3 修复；消融基础设施 |
| 2026-06-23 | v3.5 | R4 主轨 §17；P0/P1 完成；P2 启动 |
| 2026-06-24 | v3.6–v3.7 | P2 完成；F_M/F_C ES anti-overfit |
| 2026-06-25 | v3.8 | P2.5 PASS；P3 启动 |
| 2026-06-28 | v3.9–v3.10 | P3 完成；P4 MELD；tokenizer 修复 |
| 2026-06-30 | **v3.11** | **文档从 transcript 恢复**；P3 冠军更新为 M3_M7_combo；CREMA P4 log 槽位 bug 修复；P4 CREMA 重跑 |

---

*文档分工：本文 = 设计 + 工作日志；[`SDAVT_V3_R4_EXPERIMENT_RESULTS.md`](SDAVT_V3_R4_EXPERIMENT_RESULTS.md) = 实时指标表。*
