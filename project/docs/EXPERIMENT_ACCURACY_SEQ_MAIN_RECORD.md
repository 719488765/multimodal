# 准确率优化序列（AP0–AP4）主实验记录

**文档用途**：集中记录 `logs_accuracy_seq/`、`checkpoints_accuracy_seq/`、`outputs_accuracy_seq/` 中与 **AP0–AP4** 相关的实验过程、指标快照、消融对照与阶段性结论，为论文撰写提供**可溯源的数据锚点**。  
**操作规范与阶段定义**：以 `docs/EXPERIMENT_ACCURACY_SEQUENCE_GUIDE.md` 为准。  
**与历史重跑的关系**：本表数据**仅**来自 `*_accuracy_seq/` 目录；与 `logs_rerun/` 对比时须在论文中**显式标注数据来源**。

**维护约定**

- 每次重要阶段结束（或论文定稿前），在 **「修订记录」** 追加一行：日期、修订人、变更摘要。  
- **AP4 全部跑完并汇总后**，应更新本文 **§6 AP4** 的表格与 **§7 总览结论**，并可运行仓库内已有汇总脚本（若有）刷新 `outputs_accuracy_seq/accuracy_seq_results_summary.*`。  
- 指标默认从各 run 目录下的 **`metrics.csv`** 中 `phase=val` 行统计：**Last** 取最后一行 val；**Best Acc / Best F1** 取 val 行中对应列的全局最大及所在 `epoch`；**min(cls_ce_unweighted)** 用于与 ClassBalanced 尺度解耦的参考（见操作指南 §5.1）。

---

## 修订记录

| 日期 | 摘要 |
|------|------|
| 2026-05-08 | 初版：汇总当前 `logs_accuracy_seq` 内全部 run 的 `metrics.csv`；AP3 主 run 与重复目录说明；AP4 仅记录已启动的 DA 默认 run（进行中）。 |
| 2026-05-23 | **AP4 七组全部完成**：补全 §7 指标表与 §8 结论；`w005_lr5e5` 对 `checkpoint_pretrain_best*.pth` 做验证集重算；**AP3 two_stage** 根因排查：融合输出塌缩为常数、全验证集恒预测类 4。 |

---

## 1. 数据与脚本来源

| 路径（相对 `project/`） | 内容 |
|---------------------------|------|
| `logs_accuracy_seq/<run>/metrics.csv` | 主指标来源（本文表格数值快照） |
| `checkpoints_accuracy_seq/<run>/` | `checkpoint_pretrain_best.pth`、`checkpoint_pretrain_best_f1.pth` 等 |
| `outputs_accuracy_seq/accuracy_seq_results_summary.csv` | 历史导出的汇总（**可能滞后**于完整训练；以 `metrics.csv` 为准） |
| `config/rerun/accuracy_plan/` | 各阶段 YAML |

**数值快照生成方式（可复现）**：在 `project/` 下对 `logs_accuracy_seq/*/metrics.csv` 做解析：仅保留 `phase=val`，统计 Last / Best Acc@epoch / Best F1@epoch / min(cls_ce_unweighted)@epoch；同一 epoch 若有多条 val（续训重复），Best 取该 epoch 内 Acc 较高者。本文 §3–§6 基于 **2026-05-08** 解析；**§7 AP4** 基于 **2026-05-23** 解析。

---

## 2. 阶段总览与当前完成度

| 阶段 | 内容 | 当前状态（截至修订记录日期） |
|------|------|------------------------------|
| AP0 | AVT、noDA、`fusion_strategy: standard`、三混合 50 epoch | **已完成**（`AP0_*_20260413_202830`） |
| AP1 | 单数据集上界：VT×3 + AVT+ES×3 | **六条均已满 50 epoch** |
| AP2 | 三混合 + emotion_shift 配方消融（基线、M1–M4b） | **六条均已满 50 epoch**（与早期 `outputs_accuracy_seq` 中仅部分 epoch 的导出不一致时，以 `metrics.csv` 为准） |
| AP3 | 三混合融合策略：standard / leader_text / leader_audio / two_stage | **四条均已满 50 epoch**；**two_stage 训练无效**（融合输出塌缩，见 §6.3） |
| AP4 | AVT + DA 扫描（7 份 yaml） | **已全部完成**（2026-05-08～05-22，串行 `aseq_ap4_rest_serial`） |

---

## 3. AP0：三混合 noDA standard（可选对照）

| Run 目录 | Last val @epoch | Best val Acc | Best val F1 | min val cls_ce_unweighted |
|----------|-----------------|--------------|-------------|---------------------------|
| `AP0_AVT_pretrain_3datasets_noDA_standard_full50_s3407_20260413_202830` | acc=0.316415, f1=0.152107 @49 | 0.324514 @1 | 0.215163 @2 | 1.696247 @1 |

**简要解读**：与指南一致，混合 7 类验证上末段 Acc 与 F1 不高；论文若引用「无 DA standard」应对照 **Best 与 Last**，并说明与 AP2/AP3 的 **emotion_shift / 融合** 非同一管线。

---

## 4. AP1：单数据集上界（须与混合表分开展示）

**说明**：CREMA 行为 **6 类**，MELD/MOSEI 为 **7 类**；禁止与混合 7 类主表混排名而不加脚注。

### 4.1 AVT + emotion_shift（单域）

| Run | Last @49 (acc / f1) | Best Acc @ep | Best F1 @ep |
|-----|---------------------|--------------|-------------|
| `AP1_AVT_ES_pretrain_crema_only_s3407_20260420_202232` | 0.300 / 0.247 | 0.345 @42 | 0.304 @42 |
| `AP1_AVT_ES_pretrain_meld_only_s3407_20260420_202232` | 0.574 / 0.532 | 0.580 @36 | 0.540 @36 |
| `AP1_AVT_ES_pretrain_mosei_only_s3407_20260420_203623` | 0.699 / 0.594 | 0.716 @0 | 0.612 @2 |

### 4.2 VT（单域）

| Run | Last @49 (acc / f1) | Best Acc @ep | Best F1 @ep |
|-----|---------------------|--------------|-------------|
| `AP1_VT_pretrain_crema_only_s3407_20260420_202232` | 0.175 / 0.052 | 0.179 @0 | 0.054 @0 |
| `AP1_VT_pretrain_meld_only_s3407_20260420_202232` | 0.423 / 0.252 | 0.423 @0 | 0.252 @0 |
| `AP1_VT_pretrain_mosei_only_s3407_20260420_203623` | 0.716 / 0.598 | 0.716 @0 | 0.598 @0 |

**简要解读**：单域 MOSEI 上 VT 与 AVT+ES 的 Acc 均可达 **0.70+**，用于说明管线在理想条件下的上界；混合实验（AP2 起）数值较低属预期，须在论文中**分表**呈现。

---

## 5. AP2：三混合 + emotion_shift 配方消融

| Run（配置含义见指南 §3 阶段 2） | Last @49 (acc / f1) | Best Acc @ep | Best F1 @ep | min cls_ce_unweighted @ep |
|--------------------------------|---------------------|----------------|-------------|----------------------------|
| `AP2_ES_3ds_baseline_s3407_20260422_210615`（基线） | 0.462 / 0.465 | 0.591 @9 | 0.532 @12 | 1.128 @6 |
| `AP2_M1_ES_3ds_effbatch8_s3407_20260422_210615`（有效 batch 8） | 0.555 / 0.509 | 0.607 @6 | 0.562 @7 | 1.250 @0 |
| `AP2_M2_ES_3ds_lr5e5_s3407_20260422_210615`（lr 5e-5） | 0.571 / 0.536 | 0.605 @2 | 0.559 @1 | 1.633 @0 |
| `AP2_M3_ES_3ds_uniform_s3407_20260422_210615`（uniform 采样） | 0.215 / 0.255 | 0.593 @2 | 0.552 @3 | 1.459 @0 |
| `AP2_M4a_ES_3ds_plain_ce_s3407_20260422_210615`（纯 CE） | 0.180 / 0.213 | 0.580 @0 | 0.537 @4 | 1.753 @0 |
| `AP2_M4b_ES_3ds_focal_s3407_20260422_210615`（Focal） | 0.394 / 0.398 | 0.597 @3 | 0.552 @3 | 1.254 @0 |

**消融对照要点（控制变量）**：固定三数据集与 **emotion_shift** 相关协议，变动 batch / lr / 采样 / 损失形式。  
**阶段性观察**：以 **Best val Acc** 粗看，**M1（有效 batch 8）** 与 **M2（lr 5e-5）** 峰值相对较高；**M3** 末段 Acc 塌陷而早期峰值仍高，提示训练不稳定，论文中应同时报 **Best 与 Last**。最终「AP3 固定配方」若以 AP2 最优为锚，须在正文写清所选 yaml 与理由。

---

## 6. AP3：三混合融合策略消融

### 6.1 主结果 run（建议论文引用）

| 融合策略 | Run 目录 | Last @49 (acc / f1) | Best Acc @ep | Best F1 @ep | min cls_ce_unweighted @ep |
|----------|----------|---------------------|--------------|-------------|---------------------------|
| standard | `AP3_fusion_standard_3ds_s3407_20260501_110235` | 0.559 / 0.504 | 0.563 @35 | 0.518 @31 | 1.129 @24 |
| leader_follower (text) | `AP3_fusion_leader_text_3ds_s3407_20260501_110721` | 0.284 / 0.225 | 0.598 @7 | 0.542 @6 | 1.200 @0 |
| leader_follower (audio) | `AP3_fusion_leader_audio_3ds_s3407_20260501_110721` | 0.550 / 0.511 | 0.567 @43 | 0.525 @43 | 1.129 @7 |
| two_stage | `AP3_fusion_two_stage_3ds_s3407_20260501_110720` | 0.517 / 0.353 | 0.517 @0 | 0.353 @0 | 1.449 @0 | **训练无效**：见 §6.3，**勿入论文主表** |

**two_stage 说明（2026-05-23 更新）**：`metrics.csv` 虽已满 50 epoch，但为**塌缩至单一预测类**所致的平台常数，不代表有效融合；详见 §6.3。

### 6.2 已清理的失败 run（历史说明）

以下 run 曾在 `logs_accuracy_seq` 中存在，但 `metrics.csv` **仅有表头、无 `phase=val` 记录**（启动失败或未完成首轮验证）。为保持 TensorBoard 整洁，已于 **2026-05-08** 从 `logs_accuracy_seq/` 及对应空目录 `checkpoints_accuracy_seq/` 中**删除**：

- `AP3_fusion_leader_audio_3ds_s3407_20260501_110235`
- `AP3_fusion_leader_text_3ds_s3407_20260501_110235`
- `AP3_fusion_two_stage_3ds_s3407_20260501_110235`

论文中请统一引用 **§6.1** 中带完整 val 曲线的 run 名。

**消融结论（阶段性）**：在相同三混合与 seed 3407 设定下，**standard** 与 **leader_audio** 的 **Last** 表现相对接近且稳定；**leader_text** 末段 val loss 与 Acc 脱节明显（总 loss 很高），解读时应结合 **cls_ce_unweighted** 与 **Best F1**。**two_stage** 因实现/训练塌缩无效，有效融合对比仅在 standard / leader_text / leader_audio 三者间进行。

### 6.3 AP3 two_stage 根因排查（2026-05-23）

| 检查项 | 结果 |
|--------|------|
| `metrics.csv` 是否满 50 epoch val | 是（epoch 0–49） |
| 验证 Acc/F1 是否随 epoch 变化 | **否**：自 ep0 起恒为 acc=0.5173、f1=0.3528 |
| `checkpoint_pretrain_epoch_4/49.pth` 预测分布 | **400/400 样本恒为类 4**（混合 7 类标签中的 neutral） |
| `fused_features` 跨样本方差（ep4 ckpt） | **0**（所有样本融合向量完全相同） |
| 各模态 extractor 输出（ep4 ckpt） | video/audio/text 仍有 batch 间差异；**塌缩发生在 `TwoStageFusion` 内** |
| 随机初始化模型（未训练） | 融合输出仍有样本间差异（`fused_per_sample_std≈0.026`） |
| train `cls_ce_unweighted` | ep0≈1.53 → ep49≈1.58，**几乎不下降** |

**根因归纳**：`TwoStageFusion` 在 AVT（无生理信号、生理支路为零向量）+ 三混合训练下，**数轮内将融合特征压成 batch 无关常数**，分类头退化为「永远预测多数类」。`metrics.csv` 记录的是真实塌缩行为，**不是日志错误**。`models/two_stage_fusion.py` 中对多模态张量的 `view(B, num_modalities, T*hidden_dim)` 及零生理支路参与 GAT，疑为结构性诱因；**修复后须重跑 AP3 two_stage**。

---

## 7. AP4：域适应扫描（已完成）

### 7.1 计划清单（与 `ap4_da_sweep_manifest.yaml` 一致）

| 序号 | 配置文件 | Run 目录后缀 | 状态 |
|------|----------|--------------|------|
| ① | `ap4_config_AVT_DA_accuracy_seq.yaml` | `..._DA_20260508_200639` | **完成** |
| ② | `ap4_config_AVT_DA_w002_accuracy_seq.yaml` | `..._w002_20260511_193213` | **完成** |
| ③ | `ap4_config_AVT_DA_w005_accuracy_seq.yaml` | `..._w005_20260514_071550` | **完成**（5/16 自 ep24 续训） |
| ④ | `ap4_config_AVT_DA_w010_accuracy_seq.yaml` | `..._w010_20260517_063516` | **完成** |
| ⑤ | `ap4_config_AVT_DA_w005_lr5e5_accuracy_seq.yaml` | `..._w005_lr5e5_20260518_110127` | **完成** |
| ⑥ | `ap4_config_AVT_DA_uniform_accuracy_seq.yaml` | `..._uniform_20260519_154401` | **完成** |
| ⑦ | `ap4_config_AVT_DA_seed3407_accuracy_seq.yaml` | `..._seed3407_20260520_234125` | **完成**（5/22 串行末组） |

### 7.2 指标表（`metrics.csv`，2026-05-23 解析）

| 变体 | Last @49 (acc / f1) | Best Acc @ep | Best F1 @ep | min cls_ce_unweighted @ep | 备注 |
|------|---------------------|--------------|-------------|---------------------------|------|
| DA 默认 | 0.548 / 0.496 | 0.550 @45 | 0.498 @45 | 1.198 @33 | `domain_loss_weight` 默认 |
| w002 | 0.567 / 0.509 | 0.571 @47 | 0.518 @22 | 1.153 @12 | 优于默认 |
| **w005** | **0.572 / 0.528** | **0.573 @48** | **0.528 @48** | 1.130 @33 | **AP4 最佳 F1；续训后提升** |
| w010 | 0.556 / 0.498 | 0.561 @45 | 0.506 @45 | 1.186 @28 | 域权重过大，略逊于 w002/w005 |
| w005 + lr5e5 | 0.271 / 0.294 | 0.575 @31 | 0.518 @28 | 1.139 @3 | **Last 不可信**；见 §7.3 |
| uniform 采样 | 0.530 / 0.462 | 0.572 @17 | 0.517 @24 | 1.191 @3 | 早期峰值后回落 |
| seed3407 | 0.558 / 0.513 | 0.563 @40 | 0.520 @40 | 1.184 @23 | 复现性对照 |

**w005 续训说明**：ep25–27 在 `metrics.csv` 中各有 2 条 val（续训重复记录）；上表 Best 已按 epoch 内较高 Acc 去重。

### 7.3 `w005_lr5e5` 验证集重算（checkpoint 真值）

对 `AP4_AVT_pretrain_3datasets_DA_w005_lr5e5_20260518_110127` 使用 `scripts/recompute_val_metrics.py` 在全量 val 上重算：

| Checkpoint | Acc | F1 | 与 CSV Best 对照 |
|------------|-----|-----|------------------|
| `checkpoint_pretrain_best.pth` | 0.560 | 0.495 | 低于 CSV Best Acc 0.575@31 |
| `checkpoint_pretrain_best_f1.pth` | **0.567** | **0.517** | 与 CSV Best F1 0.518@28 **一致** |
| `checkpoint_pretrain_epoch_29.pth` | 0.571 | 0.498 | 接近 Acc 峰值 epoch |

**结论**：训练日志 ep44–49 的 val 断崖（Last acc≈0.27）为**评估阶段不稳定/塌缩**，不代表 best checkpoint 失效。论文报告 **w005_lr5e5** 时应以 **`checkpoint_pretrain_best_f1.pth` 重算值**（acc≈0.567，f1≈0.517）或 CSV **Best@ep31/28**，**禁止引用 Last@49**。重算 JSON：`outputs_accuracy_seq/AP4_w005_lr5e5_best_recompute.json`（`best.pth`）。

**AP4 论文选用建议**：主结果优先 **w005**（F1 最高且曲线稳）；敏感性分析覆盖 w002 / w010 / uniform / seed3407；w005_lr5e5 仅报 Best + 重算。与 AP2（emotion_shift，Best Acc≈0.61）**分表**，不可横比排名。

---

## 8. 跨阶段综合结论（2026-05-23）

1. **单域 vs 混合**：AP1 表明在单数据集上 Acc 可达较高（尤其 MOSEI）；混合三数据集（AP2 起）Acc 明显降低，须在论文中**分开展示**，避免口径混用。  
2. **配方（AP2）**：在固定 emotion_shift 三混合下，**有效 batch 与 lr** 对峰值 Acc 影响显著；**采样与损失形式** 可改变训练稳定性，末段与峰值可能背离，**禁止只报末轮**。  
3. **融合（AP3）**：**standard** 与 **leader_audio** 末段相对稳定；**leader_text** 峰值高但末段塌陷；**two_stage 当前 run 无效**（§6.3），论文融合消融不含 two_stage 直至修复重跑。  
4. **域适应（AP4）**：七组已完成。`domain_loss_weight=0.05`（w005）为 **Best F1 最优**；w002 次之；默认 DA 最低。`uniform` 与 `w005_lr5e5` 末段不稳定，须 **Best/重算** 与 **Last** 分报。AP4（standard+DA）峰值 Acc≈0.57–0.58，低于 AP2 emotion_shift≈0.61，属**不同协议**。

---

## 9. 后续维护

1. ~~补全 AP4 七组指标~~（已完成，见 §7）。  
2. **修复 `TwoStageFusion` 后重跑** `ap3_fusion_two_stage_3ds_s3407.yaml`，替换 §6.1 two_stage 行。  
3. 可选：对其余 AP4 run 的 `best_f1` checkpoint 批量 `recompute_val_metrics.py`，写入 `outputs_accuracy_seq/`。  
4. 刷新 `outputs_accuracy_seq/accuracy_seq_results_summary.csv`（若论文定稿需要），并注明导出时间与 run 目录名。

---

## 10. 参考文献（文档内）

- `docs/EXPERIMENT_ACCURACY_SEQUENCE_GUIDE.md`：阶段定义、TensorBoard、一键脚本与 **AP4 分会话手动启动（§4.4）**。  
- `docs/EXPERIMENT_RERUN_FULL_RECORD_20260407.md`：历史重跑事实锚点（**勿**与本文数值混为同一数据源）。  
- `config/rerun/accuracy_plan/ap4_da_sweep_manifest.yaml`：AP4 配置清单。
