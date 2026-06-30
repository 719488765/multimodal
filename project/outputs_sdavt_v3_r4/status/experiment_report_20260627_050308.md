# SDAVT v3 R4 论文主实验轨 — 完整实验记录

> **自动生成时间**: 2026-06-27 05:03:08  
> **刷新命令**: `bash scripts/start_sdavt_r4.sh report`  
> **进程快照**: [`SDAVT_V3_R4_EXPERIMENT_TRACKER.md`](SDAVT_V3_R4_EXPERIMENT_TRACKER.md)

---

## 1. 实验隔离与数据路径

| 资源 | 路径 | 说明 |
|------|------|------|
| 训练日志 + TensorBoard | `logs_sdavt_v3_r4/` | **论文唯一数据源** |
| Checkpoints | `checkpoints_sdavt_v3_r4/` | 与旧轨隔离 |
| 队列 / 快照 | `outputs_sdavt_v3_r4/` | experiment_queue.json, status/ |
| 汇总 CSV | `outputs_sdavt_v3_r4/tables/` | 论文表机器可读版 |
| 曲线图 | `outputs_sdavt_v3_r4/figures/` | 由 metrics.csv 导出 |
| TensorBoard | `http://<host>:6008` | `tmux attach -t sdavt_r4_tb` |

**已废弃（勿用于论文）**: `logs_sdavt_v3/`, `logs_accuracy_seq/`

### 实验阶段流水线

```
P0 fix (6) → P1 baseline (3) → P2 fusion (14) → P2.5 ES retrain → P3-M (8) + P3-C (3) → P4 modal (7) → P5 Agent
```

## 参考文献差距（Tier-2 目标）

**Tier-2 验收**：MELD Best Val **F1≥0.59** & **Acc≥0.62**（逼近 GA2MIF）；CREMA **Acc≥0.63**（人类基线）。

完整对照表：[`SDAVT_V3_REFERENCE_BENCHMARK.md`](SDAVT_V3_REFERENCE_BENCHMARK.md)

## 2. MELD（7 类，AVT 端到端）

| 方法 | WF1 | Acc | 本项目 P2 Best F1 | 本项目 P2 Best Acc | Δ WF1 |
|------|-----|-----|-------------------|--------------------|-------|
| **CFN-ESA** | 66.70% | 67.85% | — | — | 参考上界 |
| **TelME** (NAACL'24) | 67.37% | 56.70% | — | — | 参考 SOTA |
| **GA2MIF** | 58.94% | 61.65% | — | — | **Tier-2 目标** |
| **MMGCN** | 58.31% | 60.42% | — | — | |
| **emotion_shift (F_M_ES)** | — | — | **0.573** | **0.595** | **−1.6 pt vs GA2MIF** |
| standard (F_M_STD) | — | — | 0.445 | 0.497 | |
| leader_follower (F_M_LFA/LFT) | — | — | 0.421–0.456 | 0.449–0.494 | |
| two_stage (F_M_TS) | — | — | 0.368 | 0.431 | 架构简化，非 GA2MIF 级 |
| **S2-M1 历史（项目内）** | — | — | **0.605** | **0.626** | 已证可恢复 |

**Tier-2 验收**：F1 ≥ **0.59**，Acc ≥ **0.62**。


## 2. 总体进度

| Phase | 名称 | Total | Done | Running | Pending |
|-------|------|-------|------|---------|---------|
| p0_fix | P0 融合修复验证 | 6 | 6 | 0 | 0 |
| p1_baseline | P1 单域主基线 | 3 | 3 | 0 | 0 |
| p2_fusion | P2 融合策略消融 | 14 | 14 | 0 | 0 |
| p3_m3 | P3-M MELD Tier-2 | 8 | 6 | 0 | 0 |
| p3_c3 | P3-C CREMA Tier-2 | 3 | 3 | 0 | 0 |
| p4_modal | P4 模态消融 | 7 | 2 | 1 | 4 |

---

## p0_fix: P0 融合修复验证

附录 — 验证 fusion 架构修复后 6 项曾塌缩任务可正常训练

**验收标准**: train loss ≠ ln(7)；MOSEI best F1 > 0.20；非随机猜测冻结。

**修复要点** (2026-06-22): LeaderFollowerAttention Q/K/V 修正；non-ES fusion 统一 emotion_classifier；MOSEI recipe lr=1e-4。

### 任务明细

| Job | Dataset | Fusion | Status | Best Acc@ep | Best F1@ep | Epochs | Run |
|-----|---------|--------|--------|-------------|------------|--------|-----|
| F_C_TS | crema | two_stage | done | 0.2836 @ 27 | 0.2579 @ 15 | 28 | `SDAVT_R4_F_C_TS_20260622_140751` |
| F_M_STD | meld | standard | done | 0.4991 @ 2 | 0.4595 @ 4 | 13 | `SDAVT_R4_F_M_STD_20260622_140751` |
| F_M_TS | meld | two_stage | done | 0.4368 @ 9 | 0.3826 @ 9 | 18 | `SDAVT_R4_F_M_TS_20260622_163603` |
| F_O_LFT | mosei | leader_follower | done | 0.7162 @ 0 | 0.5980 @ 7 | 9 | `SDAVT_R4_F_O_LFT_20260622_214330` |
| F_O_STD | mosei | standard | done | 0.7162 @ 0 | 0.5988 @ 7 | 9 | `SDAVT_R4_F_O_STD_20260622_222125` |
| F_O_TS | mosei | two_stage | done | 0.7162 @ 0 | 0.5978 @ 0 | 9 | `SDAVT_R4_F_O_TS_20260622_232044` |

---

## p1_baseline: P1 单域主基线

Table 1 — 三数据集 Emotion Shift 主基线（R4_B_M1 / R4_B_C0 / R4_B_O0）

**目标**: MELD F1≥0.58, CREMA F1≥0.45, MOSEI F1≥0.62。

### P1 达标对照

| Job | Dataset | Best F1 | Target F1 | Best Acc | Target Acc | 达标 |
|-----|---------|---------|-----------|----------|------------|------|
| R4_B_C0 | crema | 0.5889 | 0.4500 | 0.5874 | 0.5000 | ✓ |
| R4_B_M1 | meld | 0.5680 | 0.5800 | 0.5966 | 0.6200 | △ |
| R4_B_O0 | mosei | 0.6792 | 0.6200 | 0.7269 | 0.7000 | ✓ |

### 任务明细

| Job | Dataset | Fusion | Status | Best Acc@ep | Best F1@ep | Epochs | Run |
|-----|---------|--------|--------|-------------|------------|--------|-----|
| R4_B_C0 | crema | emotion_shift | done | 0.5874 @ 19 | 0.5889 @ 19 | 32 | `SDAVT_R4_R4_B_C0_20260623_005627` |
| R4_B_M1 | meld | emotion_shift | done | 0.5966 @ 2 | 0.5680 @ 3 | 14 | `SDAVT_R4_R4_B_M1_20260623_005627` |
| R4_B_O0 | mosei | emotion_shift | done | 0.7269 @ 9 | 0.6792 @ 12 | 21 | `SDAVT_R4_R4_B_O0_20260623_032138` |

---

## p2_fusion: P2 融合策略消融

Table 2 — 5×MELD + 5×CREMA + 4×MOSEI 融合对比

### 各数据集融合冠军（P2 完成后）

| Dataset | Winner | Fusion | Best F1@ep | Best Acc@ep | Run |
|---------|--------|--------|------------|-------------|-----|
| meld | F_M_ES | emotion_shift | 0.6109 @ 3 | 0.6245 @ 3 | `SDAVT_R4_F_M_ES` |
| crema | F_C_ES | emotion_shift | 0.5412 @ 34 | 0.5470 @ 41 | `SDAVT_R4_F_C_ES` |
| mosei | F_O_ES | emotion_shift | 0.6792 @ 12 | 0.7269 @ 9 | `SDAVT_R4_F_O_ES_20260624_101647` |

### 任务明细

| Job | Dataset | Fusion | Status | Best Acc@ep | Best F1@ep | Epochs | Run |
|-----|---------|--------|--------|-------------|------------|--------|-----|
| F_C_ES | crema | emotion_shift | done | 0.5470 @ 41 | 0.5412 @ 34 | 47 | `SDAVT_R4_F_C_ES` |
| F_C_LFA | crema | leader_follower | done | 0.3065 @ 15 | 0.2978 @ 26 | 28 | `SDAVT_R4_F_C_LFA_20260624_070022` |
| F_C_LFT | crema | leader_follower | done | 0.3320 @ 15 | 0.3081 @ 15 | 28 | `SDAVT_R4_F_C_LFT_20260624_073903` |
| F_C_STD | crema | standard | done | 0.2594 @ 4 | 0.2405 @ 14 | 27 | `SDAVT_R4_F_C_STD_20260624_081739` |
| F_C_TS | crema | two_stage | done | 0.3051 @ 27 | 0.2436 @ 29 | 42 | `SDAVT_R4_F_C_TS_20260624_085905` |
| F_M_ES | meld | emotion_shift | done | 0.6245 @ 3 | 0.6109 @ 3 | 16 | `SDAVT_R4_F_M_ES` |
| F_M_LFA | meld | leader_follower | done | 0.4937 @ 1 | 0.4562 @ 3 | 12 | `SDAVT_R4_F_M_LFA_20260623_210443` |
| F_M_LFT | meld | leader_follower | done | 0.4486 @ 9 | 0.4207 @ 5 | 14 | `SDAVT_R4_F_M_LFT_20260623_225301` |
| F_M_STD | meld | standard | done | 0.4973 @ 2 | 0.4447 @ 5 | 14 | `SDAVT_R4_F_M_STD_20260624_005729` |
| F_M_TS | meld | two_stage | done | 0.4314 @ 5 | 0.3682 @ 12 | 21 | `SDAVT_R4_F_M_TS_20260624_030111` |
| F_O_ES | mosei | emotion_shift | done | 0.7269 @ 9 | 0.6792 @ 12 | 21 | `SDAVT_R4_F_O_ES_20260624_101647` |
| F_O_LFT | mosei | leader_follower | done | 0.7162 @ 0 | 0.5980 @ 7 | 9 | `SDAVT_R4_F_O_LFT_20260624_110504` |
| F_O_STD | mosei | standard | done | 0.7162 @ 0 | 0.5988 @ 7 | 9 | `SDAVT_R4_F_O_STD_20260624_115119` |
| F_O_TS | mosei | two_stage | done | 0.7162 @ 0 | 0.5978 @ 0 | 9 | `SDAVT_R4_F_O_TS_20260624_123518` |

---

## p3_m3: P3-M MELD Tier-2

Table 3 — MELD M3 优化矩阵 8 jobs（依赖 P2.5 anti-overfit ES）

**Tier-2 目标**: MELD F1≥0.59 & Acc≥0.62。

### Tier-2 达标对照

| Phase | Job | Best F1 | Target F1 | Best Acc | Target Acc | Tier-2 |
|-------|-----|---------|-----------|----------|------------|--------|
| p3_m3 | M3_M0_baseline | 0.6080 | 0.5900 | 0.6218 | 0.6200 | ✓ |
| p3_m3 | M3_M1_roberta | — | 0.5900 | — | 0.6200 | △ |
| p3_m3 | M3_M2_w2v_large | 0.5572 | 0.5900 | 0.6020 | 0.6200 | △ |
| p3_m3 | M3_M3_uniform | 0.6105 | 0.5900 | 0.6245 | 0.6200 | ✓ |
| p3_m3 | M3_M4_focal | 0.6079 | 0.5900 | 0.6209 | 0.6200 | ✓ |
| p3_m3 | M3_M5_context | 0.5725 | 0.5900 | 0.5912 | 0.6200 | △ |
| p3_m3 | M3_M6_moddrop | 0.6079 | 0.5900 | 0.6245 | 0.6200 | ✓ |
| p3_m3 | M3_M7_combo | — | 0.5900 | — | 0.6200 | △ |

### 任务明细

| Job | Dataset | Fusion | Status | Best Acc@ep | Best F1@ep | Epochs | Run |
|-----|---------|--------|--------|-------------|------------|--------|-----|
| M3_M0_baseline | meld | emotion_shift | done | 0.6218 @ 3 | 0.6080 @ 3 | 19 | `SDAVT_R4_M3_M0_baseline_20260625_200937` |
| M3_M1_roberta | meld | emotion_shift | failed | — @ — | — @ — | — | `—` |
| M3_M2_w2v_large | meld | emotion_shift | done | 0.6020 @ 3 | 0.5572 @ 3 | 19 | `SDAVT_R4_M3_M2_w2v_large_20260625_233919` |
| M3_M3_uniform | meld | emotion_shift | done | 0.6245 @ 3 | 0.6105 @ 3 | 19 | `SDAVT_R4_M3_M3_uniform_20260626_031222` |
| M3_M4_focal | meld | emotion_shift | done | 0.6209 @ 3 | 0.6079 @ 3 | 19 | `SDAVT_R4_M3_M4_focal_20260626_062916` |
| M3_M5_context | meld | emotion_shift | done | 0.5912 @ 5 | 0.5725 @ 3 | 19 | `SDAVT_R4_M3_M5_context_20260626_073046` |
| M3_M6_moddrop | meld | emotion_shift | done | 0.6245 @ 6 | 0.6079 @ 6 | 22 | `SDAVT_R4_M3_M6_moddrop_20260626_113831` |
| M3_M7_combo | meld | emotion_shift | failed | — @ — | — @ — | — | `—` |

---

## p3_c3: P3-C CREMA Tier-2

Table 3b — CREMA 补强 3 jobs（与 P3-M 后半并行）

**Tier-2 目标**: CREMA Acc≥0.63。

### Tier-2 达标对照

| Phase | Job | Best F1 | Target F1 | Best Acc | Target Acc | Tier-2 |
|-------|-----|---------|-----------|----------|------------|--------|
| p3_c3 | C3_C1_baseline | 0.5336 | — | 0.5417 | 0.6300 | △ |
| p3_c3 | C3_C2_w2v_large | 0.5629 | — | 0.5672 | 0.6300 | △ |
| p3_c3 | C3_C3_focal | 0.5526 | — | 0.5565 | 0.6300 | △ |

### 任务明细

| Job | Dataset | Fusion | Status | Best Acc@ep | Best F1@ep | Epochs | Run |
|-----|---------|--------|--------|-------------|------------|--------|-----|
| C3_C1_baseline | crema | emotion_shift | done | 0.5417 @ 29 | 0.5336 @ 29 | 45 | `SDAVT_R4_C3_C1_baseline_20260625_200937` |
| C3_C2_w2v_large | crema | emotion_shift | done | 0.5672 @ 31 | 0.5629 @ 31 | 47 | `SDAVT_R4_C3_C2_w2v_large_20260626_004150` |
| C3_C3_focal | crema | emotion_shift | done | 0.5565 @ 41 | 0.5526 @ 41 | 50 | `SDAVT_R4_C3_C3_focal_20260626_043125` |

---

## p4_modal: P4 模态消融

Table 4 — MELD 模态组合消融（依赖 P3 MELD 冠军）

### 任务明细

| Job | Dataset | Fusion | Status | Best Acc@ep | Best F1@ep | Epochs | Run |
|-----|---------|--------|--------|-------------|------------|--------|-----|
| R4_A_M_A | meld | emotion_shift | done | 0.5090 @ 19 | 0.4659 @ 31 | 47 | `SDAVT_R4_R4_A_M_A_20260626_141343` |
| R4_A_M_AT | meld | emotion_shift | pending | — @ — | — @ — | — | `—` |
| R4_A_M_AV | meld | emotion_shift | pending | — @ — | — @ — | — | `—` |
| R4_A_M_AVT | meld | emotion_shift | pending | — @ — | — @ — | — | `—` |
| R4_A_M_T | meld | emotion_shift | running | — @ — | — @ — | — | `—` |
| R4_A_M_V | meld | emotion_shift | done | 0.4377 @ 4 | 0.3840 @ 8 | 24 | `SDAVT_R4_R4_A_M_V_20260626_192132` |
| R4_A_M_VT | meld | emotion_shift | pending | — @ — | — @ — | — | `—` |

---

## 附录 A：Run 索引（TensorBoard / metrics / 曲线图）

| Phase | Job | Run | TB | metrics.csv | 曲线图 |
|-------|-----|-----|----|-------------|--------|
| p0_fix | F_C_TS | `SDAVT_R4_F_C_TS_20260622_140751` | ✓ | `logs_sdavt_v3_r4/SDAVT_R4_F_C_TS_20260622_140751/metrics.csv` | [`fig`](outputs_sdavt_v3_r4/figures/SDAVT_R4_F_C_TS_20260622_140751_curves.png) |
| p0_fix | F_M_STD | `SDAVT_R4_F_M_STD_20260622_140751` | ✓ | `logs_sdavt_v3_r4/SDAVT_R4_F_M_STD_20260622_140751/metrics.csv` | [`fig`](outputs_sdavt_v3_r4/figures/SDAVT_R4_F_M_STD_20260622_140751_curves.png) |
| p0_fix | F_M_TS | `SDAVT_R4_F_M_TS_20260622_163603` | ✓ | `logs_sdavt_v3_r4/SDAVT_R4_F_M_TS_20260622_163603/metrics.csv` | [`fig`](outputs_sdavt_v3_r4/figures/SDAVT_R4_F_M_TS_20260622_163603_curves.png) |
| p0_fix | F_O_LFT | `SDAVT_R4_F_O_LFT_20260622_214330` | ✓ | `logs_sdavt_v3_r4/SDAVT_R4_F_O_LFT_20260622_214330/metrics.csv` | [`fig`](outputs_sdavt_v3_r4/figures/SDAVT_R4_F_O_LFT_20260622_214330_curves.png) |
| p0_fix | F_O_STD | `SDAVT_R4_F_O_STD_20260622_222125` | ✓ | `logs_sdavt_v3_r4/SDAVT_R4_F_O_STD_20260622_222125/metrics.csv` | [`fig`](outputs_sdavt_v3_r4/figures/SDAVT_R4_F_O_STD_20260622_222125_curves.png) |
| p0_fix | F_O_TS | `SDAVT_R4_F_O_TS_20260622_232044` | ✓ | `logs_sdavt_v3_r4/SDAVT_R4_F_O_TS_20260622_232044/metrics.csv` | [`fig`](outputs_sdavt_v3_r4/figures/SDAVT_R4_F_O_TS_20260622_232044_curves.png) |
| p1_baseline | R4_B_C0 | `SDAVT_R4_R4_B_C0_20260623_005627` | ✓ | `logs_sdavt_v3_r4/SDAVT_R4_R4_B_C0_20260623_005627/metrics.csv` | [`fig`](outputs_sdavt_v3_r4/figures/SDAVT_R4_R4_B_C0_20260623_005627_curves.png) |
| p1_baseline | R4_B_M1 | `SDAVT_R4_R4_B_M1_20260623_005627` | ✓ | `logs_sdavt_v3_r4/SDAVT_R4_R4_B_M1_20260623_005627/metrics.csv` | [`fig`](outputs_sdavt_v3_r4/figures/SDAVT_R4_R4_B_M1_20260623_005627_curves.png) |
| p1_baseline | R4_B_O0 | `SDAVT_R4_R4_B_O0_20260623_032138` | ✓ | `logs_sdavt_v3_r4/SDAVT_R4_R4_B_O0_20260623_032138/metrics.csv` | [`fig`](outputs_sdavt_v3_r4/figures/SDAVT_R4_R4_B_O0_20260623_032138_curves.png) |
| p2_fusion | F_C_ES | `SDAVT_R4_F_C_ES` | ✓ | `logs_sdavt_v3_r4/SDAVT_R4_F_C_ES/metrics.csv` | [`fig`](outputs_sdavt_v3_r4/figures/SDAVT_R4_F_C_ES_curves.png) |
| p2_fusion | F_C_LFA | `SDAVT_R4_F_C_LFA_20260624_070022` | ✓ | `logs_sdavt_v3_r4/SDAVT_R4_F_C_LFA_20260624_070022/metrics.csv` | [`fig`](outputs_sdavt_v3_r4/figures/SDAVT_R4_F_C_LFA_20260624_070022_curves.png) |
| p2_fusion | F_C_LFT | `SDAVT_R4_F_C_LFT_20260624_073903` | ✓ | `logs_sdavt_v3_r4/SDAVT_R4_F_C_LFT_20260624_073903/metrics.csv` | [`fig`](outputs_sdavt_v3_r4/figures/SDAVT_R4_F_C_LFT_20260624_073903_curves.png) |
| p2_fusion | F_C_STD | `SDAVT_R4_F_C_STD_20260624_081739` | ✓ | `logs_sdavt_v3_r4/SDAVT_R4_F_C_STD_20260624_081739/metrics.csv` | [`fig`](outputs_sdavt_v3_r4/figures/SDAVT_R4_F_C_STD_20260624_081739_curves.png) |
| p2_fusion | F_C_TS | `SDAVT_R4_F_C_TS_20260624_085905` | ✓ | `logs_sdavt_v3_r4/SDAVT_R4_F_C_TS_20260624_085905/metrics.csv` | [`fig`](outputs_sdavt_v3_r4/figures/SDAVT_R4_F_C_TS_20260624_085905_curves.png) |
| p2_fusion | F_M_ES | `SDAVT_R4_F_M_ES` | ✓ | `logs_sdavt_v3_r4/SDAVT_R4_F_M_ES/metrics.csv` | [`fig`](outputs_sdavt_v3_r4/figures/SDAVT_R4_F_M_ES_curves.png) |
| p2_fusion | F_M_LFA | `SDAVT_R4_F_M_LFA_20260623_210443` | ✓ | `logs_sdavt_v3_r4/SDAVT_R4_F_M_LFA_20260623_210443/metrics.csv` | [`fig`](outputs_sdavt_v3_r4/figures/SDAVT_R4_F_M_LFA_20260623_210443_curves.png) |
| p2_fusion | F_M_LFT | `SDAVT_R4_F_M_LFT_20260623_225301` | ✓ | `logs_sdavt_v3_r4/SDAVT_R4_F_M_LFT_20260623_225301/metrics.csv` | [`fig`](outputs_sdavt_v3_r4/figures/SDAVT_R4_F_M_LFT_20260623_225301_curves.png) |
| p2_fusion | F_M_STD | `SDAVT_R4_F_M_STD_20260624_005729` | ✓ | `logs_sdavt_v3_r4/SDAVT_R4_F_M_STD_20260624_005729/metrics.csv` | [`fig`](outputs_sdavt_v3_r4/figures/SDAVT_R4_F_M_STD_20260624_005729_curves.png) |
| p2_fusion | F_M_TS | `SDAVT_R4_F_M_TS_20260624_030111` | ✓ | `logs_sdavt_v3_r4/SDAVT_R4_F_M_TS_20260624_030111/metrics.csv` | [`fig`](outputs_sdavt_v3_r4/figures/SDAVT_R4_F_M_TS_20260624_030111_curves.png) |
| p2_fusion | F_O_ES | `SDAVT_R4_F_O_ES_20260624_101647` | ✓ | `logs_sdavt_v3_r4/SDAVT_R4_F_O_ES_20260624_101647/metrics.csv` | [`fig`](outputs_sdavt_v3_r4/figures/SDAVT_R4_F_O_ES_20260624_101647_curves.png) |
| p2_fusion | F_O_LFT | `SDAVT_R4_F_O_LFT_20260624_110504` | ✓ | `logs_sdavt_v3_r4/SDAVT_R4_F_O_LFT_20260624_110504/metrics.csv` | [`fig`](outputs_sdavt_v3_r4/figures/SDAVT_R4_F_O_LFT_20260624_110504_curves.png) |
| p2_fusion | F_O_STD | `SDAVT_R4_F_O_STD_20260624_115119` | ✓ | `logs_sdavt_v3_r4/SDAVT_R4_F_O_STD_20260624_115119/metrics.csv` | [`fig`](outputs_sdavt_v3_r4/figures/SDAVT_R4_F_O_STD_20260624_115119_curves.png) |
| p2_fusion | F_O_TS | `SDAVT_R4_F_O_TS_20260624_123518` | ✓ | `logs_sdavt_v3_r4/SDAVT_R4_F_O_TS_20260624_123518/metrics.csv` | [`fig`](outputs_sdavt_v3_r4/figures/SDAVT_R4_F_O_TS_20260624_123518_curves.png) |
| p3_c3 | C3_C1_baseline | `SDAVT_R4_C3_C1_baseline_20260625_200937` | ✓ | `logs_sdavt_v3_r4/SDAVT_R4_C3_C1_baseline_20260625_200937/metrics.csv` | [`fig`](outputs_sdavt_v3_r4/figures/SDAVT_R4_C3_C1_baseline_20260625_200937_curves.png) |
| p3_c3 | C3_C2_w2v_large | `SDAVT_R4_C3_C2_w2v_large_20260626_004150` | ✓ | `logs_sdavt_v3_r4/SDAVT_R4_C3_C2_w2v_large_20260626_004150/metrics.csv` | [`fig`](outputs_sdavt_v3_r4/figures/SDAVT_R4_C3_C2_w2v_large_20260626_004150_curves.png) |
| p3_c3 | C3_C3_focal | `SDAVT_R4_C3_C3_focal_20260626_043125` | ✓ | `logs_sdavt_v3_r4/SDAVT_R4_C3_C3_focal_20260626_043125/metrics.csv` | [`fig`](outputs_sdavt_v3_r4/figures/SDAVT_R4_C3_C3_focal_20260626_043125_curves.png) |
| p3_m3 | M3_M0_baseline | `SDAVT_R4_M3_M0_baseline_20260625_200937` | ✓ | `logs_sdavt_v3_r4/SDAVT_R4_M3_M0_baseline_20260625_200937/metrics.csv` | [`fig`](outputs_sdavt_v3_r4/figures/SDAVT_R4_M3_M0_baseline_20260625_200937_curves.png) |
| p3_m3 | M3_M2_w2v_large | `SDAVT_R4_M3_M2_w2v_large_20260625_233919` | ✓ | `logs_sdavt_v3_r4/SDAVT_R4_M3_M2_w2v_large_20260625_233919/metrics.csv` | [`fig`](outputs_sdavt_v3_r4/figures/SDAVT_R4_M3_M2_w2v_large_20260625_233919_curves.png) |
| p3_m3 | M3_M3_uniform | `SDAVT_R4_M3_M3_uniform_20260626_031222` | ✓ | `logs_sdavt_v3_r4/SDAVT_R4_M3_M3_uniform_20260626_031222/metrics.csv` | [`fig`](outputs_sdavt_v3_r4/figures/SDAVT_R4_M3_M3_uniform_20260626_031222_curves.png) |
| p3_m3 | M3_M4_focal | `SDAVT_R4_M3_M4_focal_20260626_062916` | ✓ | `logs_sdavt_v3_r4/SDAVT_R4_M3_M4_focal_20260626_062916/metrics.csv` | [`fig`](outputs_sdavt_v3_r4/figures/SDAVT_R4_M3_M4_focal_20260626_062916_curves.png) |
| p3_m3 | M3_M5_context | `SDAVT_R4_M3_M5_context_20260626_073046` | ✓ | `logs_sdavt_v3_r4/SDAVT_R4_M3_M5_context_20260626_073046/metrics.csv` | [`fig`](outputs_sdavt_v3_r4/figures/SDAVT_R4_M3_M5_context_20260626_073046_curves.png) |
| p3_m3 | M3_M6_moddrop | `SDAVT_R4_M3_M6_moddrop_20260626_113831` | ✓ | `logs_sdavt_v3_r4/SDAVT_R4_M3_M6_moddrop_20260626_113831/metrics.csv` | [`fig`](outputs_sdavt_v3_r4/figures/SDAVT_R4_M3_M6_moddrop_20260626_113831_curves.png) |
| p4_modal | R4_A_M_A | `SDAVT_R4_R4_A_M_A_20260626_141343` | ✓ | `logs_sdavt_v3_r4/SDAVT_R4_R4_A_M_A_20260626_141343/metrics.csv` | [`fig`](outputs_sdavt_v3_r4/figures/SDAVT_R4_R4_A_M_A_20260626_141343_curves.png) |
| p4_modal | R4_A_M_V | `SDAVT_R4_R4_A_M_V_20260626_192132` | ✓ | `logs_sdavt_v3_r4/SDAVT_R4_R4_A_M_V_20260626_192132/metrics.csv` | [`fig`](outputs_sdavt_v3_r4/figures/SDAVT_R4_R4_A_M_V_20260626_192132_curves.png) |

## 附录 B：机器可读汇总

| 文件 | 内容 |
|------|------|
| `outputs_sdavt_v3_r4/tables/r4_all_runs.csv` | 全部 run 指标 |
| `outputs_sdavt_v3_r4/tables/r4_p1_baseline_table.csv` | P1 基线 |
| `outputs_sdavt_v3_r4/tables/r4_p2_fusion_table.csv` | P2 融合 |
| `outputs_sdavt_v3_r4/status/latest.json` | 最新队列快照 |
| `outputs_sdavt_v3_r4/status/p0_acceptance_latest.md` | P0 门禁验收 |

---

*本文档由 `scripts/build_sdavt_r4_report.py` 自动生成；每个 worker 任务完成后也会刷新。*
