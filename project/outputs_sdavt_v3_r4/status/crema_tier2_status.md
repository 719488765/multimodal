# CREMA Tier-2 状态（2026-07-09 close-out）

**目标**：Acc ≥ 0.63  
**当前 Champion**：**C4_C3**（`SDAVT_R4_C4_C3_c3_warmstart_acc`）  
**Best Acc**：**0.605** @ epoch 65（F1=0.606）  
**判定**：**CLOSE-OUT**（优于 C3_C2 0.567，未达 0.63，差 2.5pp）

---

## 重训轮次摘要

| 轮次 | Job | Best Acc | Best F1 | 状态 |
|------|-----|----------|---------|------|
| P3 基线 | C3_C2_w2v_large | 0.567 | 0.563 | 原 champion |
| P3-C+ R1 | C4_C1_combo_acc | — | — | failed_nan |
| P3-C+ R2 | C4_C2_c3_base_acc | 0.353 | 0.315 | completed_not_met |
| **P3-C+ R3** | **C4_C3_c3_warmstart_acc** | **0.605** | **0.606** | **PARTIAL**（+3.8pp vs C3_C2） |

---

## 决策

- **不再开第 4 轮 GPU 重训**（ROI 低；C4_C1/C4_C2 已证明激进改 recipe 会退化）
- **论文固定**：C4_C3 Acc=0.605
- **可选非 GPU 工作**（单独立项）：ensemble C4_C3 + C3_C2；roberta-large 文本骨干

---

## 指标来源

- CSV：`logs_sdavt_v3_r4/SDAVT_R4_C4_C3_c3_warmstart_acc/metrics.csv`
- 快照：`r4_parallel_retrain_final_latest.json`
- TensorBoard：`logs_sdavt_v3_r4/` @ port 6008
