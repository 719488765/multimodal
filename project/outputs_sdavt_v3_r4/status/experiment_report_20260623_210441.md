# SDAVT v3 R4 论文主实验轨 — 完整实验记录

> **自动生成时间**: 2026-06-23 21:04:41  
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
P0 fix (6) → P1 baseline (3) → P2 fusion (14) → P3 M3 (6) → P4 modal (7)
```

## 2. 总体进度

| Phase | 名称 | Total | Done | Running | Pending |
|-------|------|-------|------|---------|---------|
| p0_fix | P0 融合修复验证 | 6 | 6 | 0 | 0 |
| p1_baseline | P1 单域主基线 | 3 | 3 | 0 | 0 |
| p2_fusion | P2 融合策略消融 | 14 | 1 | 0 | 13 |
| p3_m3 | P3 MELD 拉升 | 0 | 0 | 0 | 0 |
| p4_modal | P4 模态消融 | 0 | 0 | 0 | 0 |

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
| meld | F_M_ES | emotion_shift | 0.5731 @ 3 | 0.5948 @ 2 | `SDAVT_R4_F_M_ES_20260623_193605` |
| crema | — | — | — | — | — |
| mosei | — | — | — | — | — |

### 任务明细

| Job | Dataset | Fusion | Status | Best Acc@ep | Best F1@ep | Epochs | Run |
|-----|---------|--------|--------|-------------|------------|--------|-----|
| F_C_ES | crema | emotion_shift | pending | — @ — | — @ — | — | `—` |
| F_C_LFA | crema | leader_follower | pending | — @ — | — @ — | — | `—` |
| F_C_LFT | crema | leader_follower | pending | — @ — | — @ — | — | `—` |
| F_C_STD | crema | standard | pending | — @ — | — @ — | — | `—` |
| F_C_TS | crema | two_stage | pending | — @ — | — @ — | — | `—` |
| F_M_ES | meld | emotion_shift | done | 0.5948 @ 2 | 0.5731 @ 3 | 9 | `SDAVT_R4_F_M_ES_20260623_193605` |
| F_M_LFA | meld | leader_follower | pending | — @ — | — @ — | — | `—` |
| F_M_LFT | meld | leader_follower | pending | — @ — | — @ — | — | `—` |
| F_M_STD | meld | standard | pending | — @ — | — @ — | — | `—` |
| F_M_TS | meld | two_stage | pending | — @ — | — @ — | — | `—` |
| F_O_ES | mosei | emotion_shift | pending | — @ — | — @ — | — | `—` |
| F_O_LFT | mosei | leader_follower | pending | — @ — | — @ — | — | `—` |
| F_O_STD | mosei | standard | pending | — @ — | — @ — | — | `—` |
| F_O_TS | mosei | two_stage | pending | — @ — | — @ — | — | `—` |

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
| p2_fusion | F_M_ES | `SDAVT_R4_F_M_ES_20260623_193605` | ✓ | `logs_sdavt_v3_r4/SDAVT_R4_F_M_ES_20260623_193605/metrics.csv` | [`fig`](outputs_sdavt_v3_r4/figures/SDAVT_R4_F_M_ES_20260623_193605_curves.png) |

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
