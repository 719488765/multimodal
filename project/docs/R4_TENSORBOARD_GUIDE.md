# R4 TensorBoard 使用与清理说明

**访问地址：** [http://127.0.0.1:6008](http://127.0.0.1:6008) → Scalars / Timeseries  
**logdir：** `project/logs_sdavt_v3_r4/`（**57** 条有效 run，2026-07-09 清理后）

---

## 1. 清理结果（2026-07-09）

已从 TensorBoard 视图中移除 **8** 条无效/重复 run，移至：

```text
project/logs_sdavt_v3_r4_archived/
```

| 类别 | Run 数 | 说明 |
|------|--------|------|
| **failed_nan** | 1 | `C4_C1_combo_acc` — 训练 NaN，无 val 曲线 |
| **MOSEI 重试** | 6 | P0 阶段 aborted retry（F1≈0.088）；canonical run 已保留 |
| **空 metrics** | 1 | `M3_M7_chinese_agent` — finetune 未写出有效 epoch |

完整清单：`outputs_sdavt_v3_r4/status/tb_cleanup_manifest_20260709.json`

---

## 2. 保留的 57 条 run 构成

| 来源 | 数量 |
|------|------|
| R4 队列（55 jobs） | 55 |
| Close-out CREMA（C4_C2/C4_C3） | 2 |
| **合计** | **57** |

含 P0 与 P2 同名 job（如 `F_C_TS`）的**两阶段合法 run** — 均已保留。

---

## 3. 常用操作

```bash
# 启动 / 重启 TensorBoard
bash scripts/tensorboard_sdavt_r4.sh 6008
tmux attach -t sdavt_r4_tensorboard

# 甄别并归档无效 run（默认 dry-run）
python scripts/prune_r4_tensorboard_logs.py
python scripts/prune_r4_tensorboard_logs.py --apply

# 恢复误归档 run
mv logs_sdavt_v3_r4_archived/<run_dir> logs_sdavt_v3_r4/
```

---

## 4. TensorBoard 浏览建议

1. **Filter runs**：搜索 `M3_M7`、`C4_C3`、`F_O_ES`、`R4_A_M` 等 job 前缀  
2. **Scalars**：`val/f1`、`val/accuracy`、`train/loss`  
3. **P4 模态对比**：筛选 `R4_A_M_`（MELD）/ `R4_A_O_`（MOSEI）/ `R4_A_C_`（CREMA）  
4. **避免混淆**：旧轨 AP 实验在 `logs_accuracy_seq/`（端口 **6007**），勿与 R4 混看

---

## 5. 关于「失败但保留」的 run

以下 run **训练有效**，曲线保留在 TB 中，但指标较低或为 collapse，**不是数据错误**：

| 示例 | 说明 |
|------|------|
| `R4_A_M_V` | MELD 视频单模态 F1≈0.27（任务固有难度） |
| `R4_A_C_T` / `R4_A_C_A` | CREMA 单模态弱基线 |
| P2 `F_C_STD` / `F_C_LFA` | 融合消融对照组 |

详见 [`R4_FULL_EXPERIMENT_REPORT.md`](R4_FULL_EXPERIMENT_REPORT.md) §7–§8。
