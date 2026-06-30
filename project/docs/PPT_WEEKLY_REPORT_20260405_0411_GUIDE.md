# 工作总结汇报 PPT 编写指导（4月5日–4月11日）

编写说明：版式建议与 `PPT_WEEKLY_REPORT_20260329_0404_GUIDE.md` 保持一致。本周**主线成果**为专用目录下 **6 组重跑实验全部闭环**（数据以 `logs_rerun` 与 `outputs_rerun/rerun_results_summary` 为准）。`EXPERIMENT_ACCURACY_SEQUENCE_GUIDE.md` 与 `EMOTION_AGENT_ENGINEERING_PLAN.md` 为**后续计划与构想**，本周以文档与工程准备为主，**新实验序列尚未开跑**。汇报时建议区分「本周已完成」与「下周及后续」。

---

## 第 1 页：封面

- **标题**：工作总结汇报  
- **副标题**：李智春 | **2026年4月5日–4月11日**  
- **可选副标题**：主线重跑六组闭环 · 融合与模态消融结论沉淀 · 准确率优化序列与智能体方案就绪（待执行）

---

## 第 2 页：本周工作概述（一页说清）

**已完成（实验主线）**

- 在 **`logs_rerun` / `checkpoints_rerun`** 隔离环境下，完成 **6 组**预训练重跑：**AT±DA、VT noDA、AVT±DA、AVT noDA（standard）、AVT noDA（emotion_shift）**。  
- 其中 **5 组满 50 个验证 epoch**；**AVT noDA（standard）** 验证记录为 **42 条（epoch 0–41）**，与满程 run 比「末轮」需加脚注。  
- 指标已汇总至 `outputs_rerun/rerun_results_summary`，并与 `EXPERIMENT_RERUN_FULL_RECORD_20260407.md` 对齐。

**本周工程与文档（支撑论文与下一轮实验）**

- 训练侧增强验证 **损失分项与不加权 CE** 落盘（`metrics.csv` / TensorBoard），避免 val `cls_loss` 恒为 0 的误判。  
- **预训练按配置过滤单数据集**（与 finetune 一致），支撑后续单域上界实验。  
- 新建 **`accuracy_plan`** 配置族、**`logs_accuracy_seq`** 等新目录约定，并撰写 **`EXPERIMENT_ACCURACY_SEQUENCE_GUIDE.md`**（分阶段配方 / 融合 / DA 扫描与论文记录规范）。  
- 一键 **tmux / 双卡** 启动脚本（`start_accuracy_seq_*.sh`）、TensorBoard 指向脚本。  
- **情感智能体工程方案**成文：`EMOTION_AGENT_ENGINEERING_PLAN.md`（采集→推理→LLM→UI，与训练解耦）。

**未启动（明确写进「后续」）**

- **准确率优化实验序列**（`logs_accuracy_seq`）本周**未开跑**，仅完成配置与操作指南。  
- **情感智能体**仍为方案级，无在线服务落地。

---

## 第 3 页：时间线（4月5日–11日，可按实际微调）

| 时段 | 内容概要 |
|------|----------|
| 本周初–中期 | 持续推进 / 收尾 **AVT noDA（standard）**、**AVT+emotion_shift** 等三模态重跑；与 tmux 会话 `rerun_avt_noda`、`rerun_avt_es` 等对齐。 |
| 4月7日前后 | 多组重跑时间戳目录落盘（如 `20260407_193400` 批次），与 4月3日批次共同构成当前 `logs_rerun` 快照。 |
| 本周后期 | 全量记录写入 `EXPERIMENT_RERUN_FULL_RECORD_20260407.md`（v2）；汇总表与原始 `metrics.csv` 一致性校验。 |
| 4月11日 | 定稿 **准确率优化序列**操作指南与启动脚本；**情感智能体**工程方案文档；本周汇报材料定稿。 |

PPT 口述要点：上周以「排队与短程」为主，**本周主线是六组重跑闭环 + 结论与风险写清 + 下一轮实验与系统方案铺好路**。

---

## 第 4 页：六组重跑执行状态（与 logs_rerun 一致）

| 实验设定 | 验证条数 | 状态说明 |
|----------|----------|----------|
| AT + noDA | 50（0–49） | 已完成 |
| AT + DA | 50 | 已完成 |
| VT + noDA | 50 | 已完成 |
| AVT + DA | 50 | 已完成 |
| AVT + noDA（standard） | **42（0–41）** | **未满 50 epoch**，末段 Acc 塌陷，但存在 **Best@epoch27** |
| AVT + noDA（emotion_shift） | 50 | 已完成；**末轮 val loss 数量级异常**，Acc 仍约 0.37 |

---

## 第 5 页：核心数值表（建议直接做进 PPT 表）

**说明**：下列数字来自 `rerun_results_summary` / 全量记录文档，汇报时建议**同时带「末轮」与「最佳（epoch）」**两列，避免 DA / standard 仅报末轮造成误判。

| 设定 | Last Acc / F1 | Best Acc（epoch） | Best F1（epoch） |
|------|----------------|-------------------|------------------|
| AT noDA | 0.255 / 0.228 | 0.322（19） | 0.254（27） |
| AT DA | 0.182 / 0.134 | **0.353（24）** | **0.313（24）** |
| VT noDA | **0.374** / **0.341** | 0.380（45） | 0.347（45） |
| AVT DA | 0.276 / 0.226 | 0.354（42） | 0.271（42） |
| AVT noDA standard | 0.171 / 0.130 | 0.373（27） | 0.329（27） |
| AVT noDA **emotion_shift** | 0.375 / 0.300 | **0.445（19）** | **0.404（19）** |

**一句话结论（可放在本页脚注）**

- **峰值最强**：AVT + emotion_shift（Best Acc/F1 本轮最高）。  
- **末轮较稳**：VT noDA（末轮与最佳相对均衡）。  
- **DA**：AT 上峰值优于 noDA，但**末轮差于 noDA**——论文需写清 Best vs Last 与早停策略。

---

## 第 6 页：消融结论（论文叙事可用）

1. **模态**：VT 相对 AT，**末轮 Acc 提升约 +0.12**；引入三模态 + DA **未超过** VT noDA 末轮表现（复杂度↑未必收益↑）。  
2. **域适应（AT）**：**验证峰值提升**，**末轮下滑**——适合作为「有效但不稳」的消融，不宜只报 last。  
3. **融合（AVT noDA）**：**emotion_shift** 在 **Best Acc/F1** 与 **末轮 Acc** 上均优于同配置 **standard**，支撑 CFN-ESA 类融合叙事；standard 需补满程或统一截断再比 last。  
4. **风险**：emotion_shift **末轮 val loss ≈ 56.7** 与 AT/VT 量级不一致——写作时以 **Acc/F1 与 checkpoint 重算** 为准，loss 曲线需结合分项 / 不加权 CE 解释（见下周序列）。

---

## 第 7 页：问题与风险（诚实写，体现严谨）

| 问题 | 影响 | 已采取 / 计划动作 |
|------|------|-------------------|
| AVT standard 仅 42 条 val | 与满 50 epoch 比 last **不公平** | 论文报 **Best@27**；后续 **AP0 补跑满程**（`accuracy_plan`） |
| emotion_shift 末轮 val loss 异常高 | 易误读为「训练崩坏」 | 代码侧已加强 **val 分项与 cls_ce_unweighted**；定稿用 **recompute_val_metrics** 交叉验证 |
| 混合 7 类 Acc 绝对值不高 | 域偏移 + 标签对齐天花板 | 单数据集上界实验（AP1）与混合实验 **分表** 报告 |
| ClassBalanced 按 batch 重算 | val total loss 与 Acc 可脱钩 | 后续序列统一监控 **不加权 CE** 与 **固定类权重** 可选配置 |

---

## 第 8 页：本周工程产出清单（非实验数字，体现工作量）

- 重跑全量记录与结论：**`EXPERIMENT_RERUN_FULL_RECORD_20260407.md`（v2）**。  
- 训练脚本：**验证损失分项、不加权 CE、预训练数据集过滤、ClassBalanced 固定权重 / label smoothing / 骨干分组 LR（可选）**。  
- 下一轮实验包：**`config/rerun/accuracy_plan/*.yaml`** + **`EXPERIMENT_ACCURACY_SEQUENCE_GUIDE.md`** + **`start_accuracy_seq_tmux.sh` / `start_accuracy_seq_gpuaware.sh` / `tensorboard_accuracy_seq.sh`**。  
- 新目录约定：**`logs_accuracy_seq`**、**`checkpoints_accuracy_seq`**（与本周 **`logs_rerun`** 隔离）。  
- 论文配套工程蓝图：**`EMOTION_AGENT_ENGINEERING_PLAN.md`**（实时采集、推理服务、LLM 编排、UI）。

---

## 第 9 页：后续工作计划（准确率优化序列，未开跑）

**定位**：解决本周暴露的 **batch 过小、融合消融不全、loss 可比性、DA 末轮不稳** 等问题，目标 **提升混合训练下可复现的验证表现与论文表格完备度**。

**阶段概要（详见操作指南）**

- **AP0**：AVT standard **满 50 epoch + seed 3407**。  
- **AP1**：单数据集上界（VT / AVT+ES × CREMA·MELD·MOSEI）。  
- **AP2**：三混合上 **配方消融**（有效 batch、lr、uniform、CE/Focal、可选 fixed CB 等）。  
- **AP3**：**融合消融**（standard、leader_follower、two_stage；emotion_shift 由 AP2 基线对照）。  
- **AP4**：**DA 权重扫描**（配置已复制到 `accuracy_plan`，日志进 `logs_accuracy_seq`）。

**论文记录规范（可在 PPT 用一条呈现）**：固定协议下 **控制变量**；主表报 **Best F1 / Best Acc + epoch**，DA 必报 **Last**；单混合 **分表**。

---

## 第 10 页：后续工作计划（情感智能体，方案级）

**目标**：在现有 **`inference.py` + 多模态模型** 之上，落地 **采集 → 三模态滑窗 → 情绪识别 → LLM 话术 → UI**，与 **训练主链路解耦**。

**首版范围（来自工程方案）**

- 视频 / 音频 / ASR 文本；滑窗 3s、步长 1s 等建议参数。  
- 推理服务化、LLM 编排、Web UI、日志归档；**不修改 `train.py` 主逻辑**。

**本周状态**：文档与架构图已就绪；**未进入开发迭代**，可与论文实验章节并行排期。

---

## 第 11 页：经验小结与致谢（可选）

- **可复现**：专用 `logs_rerun` 目录 + 统一命名 + 汇总脚本，便于组内对齐与论文溯源。  
- **指标口径**：修复后链路下，**务必 Best/Last 齐报**，避免 DA 与末段过拟合叙事冲突。  
- **下一步**：执行 **accuracy_seq** 序列时，优先 **分阶段 tmux**，避免 AVT 多任务同卡 OOM。

---

## 附录：PPT 页序建议（共约 11 页）

1. 封面  
2. 本周概述（完成 / 文档 / 未启动）  
3. 时间线（4/5–4/11）  
4. 六组执行状态  
5. 核心数值表（Last + Best）  
6. 消融结论（4 条 bullet）  
7. 问题与风险表  
8. 工程产出清单  
9. 后续：准确率优化序列  
10. 后续：情感智能体方案  
11. 小结（可选）

---

**文档版本**：v1  
**数据快照**：`project/logs_rerun/`、`project/outputs_rerun/rerun_results_summary.md`（与 `EXPERIMENT_RERUN_FULL_RECORD_20260407.md` v2 一致）  
**关联文档**：`EXPERIMENT_RERUN_FULL_RECORD_20260407.md`、`EXPERIMENT_ACCURACY_SEQUENCE_GUIDE.md`、`EMOTION_AGENT_ENGINEERING_PLAN.md`
