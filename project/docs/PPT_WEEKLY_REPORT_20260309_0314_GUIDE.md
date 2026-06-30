# 工作总结汇报 PPT 编写指导（3月9日–3月14日）

**编写说明**：本文档依据 `PROJECT_OVERVIEW_AND_TRAINING_PLAN.md` 第十节工作日志与实验评价，梳理本周（2026-03-09 至 2026-03-14）工作内容，并给出 PPT 页结构、每页要点及**实验截图占位**，便于按上周汇报格式完成本周 PPT。图片以**文件名占位**，插入 PPT 时替换为实际图片即可。

**使用方式**：各页下的「**PPT 可直接使用文案**」或「**策略阐述**」可直接复制到对应幻灯片作为标题/正文/结论；表格可直接粘贴到 PPT 表格中。

---

## 一、本周工作范围与时间线

| 日期 | 内容概要 |
|------|----------|
| 3月9日前后 | 数据清洗脚本与目录结构版检测/归档、TensorBoard 与日志命名规范 |
| 3月11日 | T-only / A-only 单模态预训练完成（文档 10.5.1 / 10.5.2）|
| 3月12日 | AT + 域适应（DA）预训练完成（文档 10.5.3）|
| 3月13–14日 | CREMA-D 预训练+微调 vs 从零训练完成并记录（文档 10.5.4），日志重命名为 AT_crema_from_pretrain_noDA_* / AT_crema_scratch_noDA_* |

**PPT 可直接使用（时间线 bullet，可选单独一页或放在概述页）**  
- 3/9 前后：数据清洗与日志命名规范  
- 3/11：T-only、A-only 单模态预训练（模态消融）  
- 3/12：AT+DA 预训练（域适应消融）  
- 3/13–14：CREMA 预训练+微调 vs 从零训练（预训练收益对比）

---

## 二、PPT 整体结构建议（对齐上周格式）

建议页序：**封面 → 本周工作概述 → 实验策略与消融设计 → 训练/验证曲线（配图）→ CREMA 小结表 → 消融总结 → 下周计划 → 致谢（可选）**。

---

## 三、各页内容与图片占位设计

### 第 1 页：封面

- **标题**：工作总结汇报
- **副标题**：李智春 | 2026年3月9日–3月14日
- **图片**：无

**PPT 可直接使用文案**  
- 主标题：工作总结汇报  
- 副标题：李智春 | 2026年3月9日–3月14日  
- 可选副标题：多模态情绪识别实验进展 / 预训练与 CREMA 微调对比

### 第 2 页：本周工作概述

- **标题**：本周工作概述（3.9–3.14）
- **要点**：单模态消融 T-only/A-only（3/11）；域适应消融 AT+DA（3/12）；CREMA 微调 vs scratch（3/13–14）；数据清洗脚本、微调配置与训练脚本更新。
- **图片**：无

**PPT 可直接使用文案（建议分 4 条 bullet）**  
- 完成**单模态消融**：T-only、A-only 三数据集预训练（3/11），量化文本/音频各自贡献。  
- 完成**域适应消融**：AT + DA 三数据集预训练（3/12），对比有/无域适应模块的效果。  
- 完成**CREMA-D 微调对比**：预训练+微调 vs 从零训练（3/13–14），验证预训练对单数据集泛化的收益。  
- **工程与数据**：目录结构版数据清洗脚本、微调/scratch 配置与训练脚本（finetune 过滤、resume 兼容）更新。

---

### 第 3 页：实验策略与消融设计

- **标题**：实验策略与消融设计
- **图片**：无

**策略阐述（可直接作 PPT 小标题 + 正文）**

1. **模态消融（Modality Ablation）**  
   - **目的**：在相同训练设定下，分别只用文本（T-only）、只用音频（A-only），与双模态 AT 基线对比，量化各模态对三数据集混合预训练的贡献。  
   - **做法**：保持 50 epoch、batch_size=4、lr=1e-4 等与 AT Baseline 一致，仅关闭另一模态；域适应关闭（noDA）。  
   - **对比对象**：AT Baseline（10.5） vs T-only（10.5.1） vs A-only（10.5.2）。

2. **域适应消融（Domain Adaptation Ablation）**  
   - **目的**：考察在三数据集混合预训练中，引入域适应模块（DA）是否能在验证集上带来稳定提升。  
   - **做法**：同 AT 双模态、同超参，仅开启 `domain_adaptation.enabled=true` 与 `domain_loss_weight=0.1`，训练 50 epoch。  
   - **对比对象**：AT noDA（10.5） vs AT+DA（10.5.3）。

3. **预训练 vs 从零训练（Pretrain+Finetune vs Scratch）**  
   - **目的**：验证「三数据集预训练 → 单数据集（CREMA-D）微调」是否优于「仅在 CREMA-D 上从零训练」，为预训练收益提供证据。  
   - **做法**：一组从 `checkpoint_pretrain_best.pth` 加载后在 CREMA 上微调 30 epoch；另一组随机初始化、同一 CREMA 数据与 30 epoch。  
   - **对比对象**：AT_crema_from_pretrain_noDA_* vs AT_crema_scratch_noDA_*（10.5.4）。

**PPT 可直接使用文案（精简版，每块 1 标题 + 2–3 句）**  
- **模态消融**：T-only / A-only 与 AT 同配置预训练 50 epoch，对比训练损失与验证 Accuracy/F1，量化文本、音频各自贡献。  
- **域适应消融**：AT+DA 与 AT noDA 同配置预训练 50 epoch，对比验证指标，评估当前 DA 模块的收益。  
- **预训练 vs 从零**：预训练权重在 CREMA 上微调 30 epoch vs 随机初始化在 CREMA 上训练 30 epoch，对比验证 F1 与收敛表现。

### 第 4 页：训练分类损失对比（train/loss_classification）

- **标题**：训练分类损失对比（train/loss_classification）
- **说明**：预训练+微调（绿）起点约 2.55、收敛至约 2.24；scratch（橙）起点约 2.27、收敛至约 2.23；T-only 最低约 2.11，A-only 最高约 2.51。
- **图片占位（主图）**：`train-loss_classification-c551afe9-a811-4c4f-a8dc-524a17626d8c.png`
- **图片占位（备选）**：`train-loss_classification_timeseries-ca174866-21e2-4abe-92de-c4015c8ba131.png`
- **版式建议**：图居中或偏左，约占页宽 65%–70%；右侧或下方标注绿=预训练+微调、橙=scratch。

**PPT 可直接使用文案**  
- 绿线：预训练+CREMA 微调（Step 29，1.687 hr）；橙线：CREMA 从零训练（Step 29，31.72 min）。  
- 紫/蓝/绿/橙长线：AT noDA、AT+DA、T-only、A-only 三数据集预训练（Step 49）。T-only 收敛最低（~2.11），A-only 最高（~2.51）。

---

### 第 5 页：训练总损失对比（train/loss_total）

- **标题**：训练总损失对比（train/loss_total）
- **图片占位（主图）**：`train-loss_total-37a81085-df5e-4707-b81f-fc3663b0cd09.png`
- **图片占位（备选）**：`train-loss_total_timeseries-b3be72fd-7cac-4c82-9c33-99d5cde63440.png`

**PPT 可直接使用文案**  
- 与第 4 页 run 对应关系一致；两条 CREMA run 均在约 29 step 结束，预训练 run 为 49 step。

---

### 第 6 页：验证准确率对比（val/accuracy）

- **标题**：验证准确率对比（val/accuracy）
- **说明**：CREMA 两条 run 约 0.17–0.19，高于多数三数据集预训练 run。
- **图片占位（主图）**：`val-accuracy-3ebb46aa-54bc-4a72-88b2-699f8af24fe7.png`
- **图片占位（备选）**：`val-accuracy_timeseries-e185e7e8-6606-4356-b986-72e8f61cdd75.png`

**PPT 可直接使用文案**  
- CREMA 微调/scratch 验证准确率约 0.17–0.19，高于三数据集预训练基线（AT noDA/DA、T-only 约 0.13–0.16；A-only 约 0.08）。  
- 说明在目标单数据集上微调或专注训练能显著提升验证准确率。

---

### 第 7 页：验证 F1 对比（val/f1）— 核心结论页

- **标题**：验证 F1 对比（val/f1）— CREMA 微调 vs 从零训练
- **说明**：预训练+微调（绿）val/f1 约 **0.2977**，scratch（橙）约 **0.2265**；预训练+微调明显优于从零训练。
- **图片占位**：`val-f1_timeseries-b2ec3e0b-9dab-4193-9324-40a5fc0a6e81.png`
- **版式建议**：此页为重点，图可适当放大，配合 1–2 句结论文字。

**PPT 可直接使用文案（结论句建议放在图下方或右侧）**  
- **预训练+微调**（绿）：val/f1 ≈ **0.30**（Step 29）；**从零训练**（橙）：val/f1 ≈ **0.23**（Step 29）。  
- **结论**：三数据集预训练后再在 CREMA-D 上微调，验证 F1 明显优于仅在 CREMA 上从零训练，预训练带来稳定泛化收益。

---

### 第 8 页：验证总损失对比（val/loss_total）

- **标题**：验证总损失对比（val/loss_total）
- **说明**：预训练+微调约 3.88，scratch 约 4.22；预训练+微调更低更稳定。
- **图片占位（主图）**：`val-loss_total-0140d668-3384-4ae6-b907-c8b5adef937e.png`
- **图片占位（备选）**：`val-loss_total_timeseries-000041cb-d4e9-413f-8531-37c52e68ae64.png`

**PPT 可直接使用文案**  
- 预训练+微调 val/loss_total ≈ 3.88，scratch ≈ 4.22；预训练+微调验证损失更低、曲线更稳定。  
- A-only 预训练验证损失约 8.26，明显发散；CREMA 两条 run 收敛在 3.9–4.2 区间。

---

### 第 9 页：验证 Precision / Recall（可选）

- **标题**：验证 Precision 与 Recall
- **左图占位**：`val-precision-9ebd9aae-33de-4d6a-99a3-1f8b610eb85a.png` 或 `val-precision_timeseries-391e772b-6aff-4210-b49f-bf1c850f61cd.png`
- **右图占位**：`val-recall-0349130d-9e7d-47b9-83f2-51ed1f839fd6.png`

**PPT 可直接使用文案**  
- 预训练+微调 val/precision 约 0.995（Step 29），scratch 约 0.32；val/recall 两条 run 均在约 0.17–0.19 区间，可作为 F1 的补充说明。

### 第 10 页：CREMA 微调 vs 从零训练 — 小结表

- **标题**：CREMA-D 微调 vs 从零训练 — 小结
- **图片**：无（或复用第 7 页 val/f1 图缩小置于一侧）

**PPT 可直接使用表格（复制到 PPT 表格中）**

| 实验 | 运行目录（示例） | 训练时长 | 最终 train loss | val/accuracy | val/f1 |
|------|------------------|----------|------------------|--------------|--------|
| 预训练+微调 | AT_crema_from_pretrain_noDA_20260313_212316 | 1.687 hr | ~2.24 | ~0.176 | **~0.30** |
| 从零训练(scratch) | AT_crema_scratch_noDA_20260314_102323 | 31.72 min | ~2.23 | ~0.187 | ~0.23 |

**PPT 可直接使用结论句**  
- 预训练+微调在 CREMA-D 验证集上 **val/f1 达 ~0.30**，**优于从零训练的 ~0.23**，说明三数据集预训练后再在单数据集上微调能带来明确的泛化收益。  
- 训练墙钟时间 scratch 更短（31.72 min vs 1.687 hr），可能受环境因素影响；以验证指标为主要对比依据。

---

### 第 11 页：消融实验对比总结

- **标题**：消融实验对比总结（3.9–3.14）
- **图片**：无；或放 val/f1 图小图。

**PPT 可直接使用文案（三条结论，每条可作一 bullet）**  
- **模态消融**：AT（音频+文本）> T-only（文本）≈ A-only（音频）。文本模态在三数据集混合场景下更稳定，音频单独预训练效果有限，与文本结合后贡献明显。  
- **域适应消融**：AT+DA 与 AT noDA 在验证 Accuracy/F1 上相近，当前 DA 模块未带来稳定提升；后续可针对结构或超参进一步调优。  
- **预训练 vs 从零**：在 CREMA-D 上，预训练+微调 val/f1（~0.30）优于从零训练（~0.23），建议以「预训练+单数据集微调」为默认方案。

---

### 第 12 页：下周计划

- **标题**：下周计划
- **图片**：无

**PPT 可直接使用文案（3–4 条 bullet）**  
- 可选：**AT+DA 预训练 → CREMA 微调**，与当前 noDA 微调对比，评估域适应在 downstream 上的作用。  
- 可选：CREMA 上**多 seed 或统一 early stopping**，强化预训练 vs scratch 结论的稳定性。  
- **论文**：将 10.5.1–10.5.4 结果整理入「模态消融」「域适应消融」「预训练收益」表格与曲线，对齐 9.5 基线结果表。

### 第 13 页：致谢 / 附录（可选）

- **图片**：无

---

## 四、图片文件名与用途速查表

| 页面 | 主图文件名 | 备选文件名 |
|------|------------|------------|
| 第 4 页 | train-loss_classification-c551afe9-a811-4c4f-a8dc-524a17626d8c.png | train-loss_classification_timeseries-ca174866-21e2-4abe-92de-c4015c8ba131.png |
| 第 5 页 | train-loss_total-37a81085-df5e-4707-b81f-fc3663b0cd09.png | train-loss_total_timeseries-b3be72fd-7cac-4c82-9c33-99d5cde63440.png |
| 第 6 页 | val-accuracy-3ebb46aa-54bc-4a72-88b2-699f8af24fe7.png | val-accuracy_timeseries-e185e7e8-6606-4356-b986-72e8f61cdd75.png |
| 第 7 页 | val-f1_timeseries-b2ec3e0b-9dab-4193-9324-40a5fc0a6e81.png | — |
| 第 8 页 | val-loss_total-0140d668-3384-4ae6-b907-c8b5adef937e.png | val-loss_total_timeseries-000041cb-d4e9-413f-8531-37c52e68ae64.png |
| 第 9 页 | val-precision-9ebd9aae-33de-4d6a-99a3-1f8b610eb85a.png，val-recall-0349130d-9e7d-47b9-83f2-51ed1f839fd6.png | val-precision_timeseries-391e772b-6aff-4210-b49f-bf1c850f61cd.png |

---

## 五、编写时注意事项

1. **与上周格式对齐**：若上周有固定版式（标题栏、logo、页码），本周沿用；页标题与 bullets 层级与上周保持一致。
2. **数据一致性**：PPT 中数值以 `PROJECT_OVERVIEW_AND_TRAINING_PLAN.md` 第十节为准。
3. **图片替换**：插入 PPT 时按上表文件名或本地实际路径插入；主图与时间序列版择一即可，避免重复。
4. **核心结论突出**：第 7 页（val/f1）和第 10 页（小结表）是「预训练+微调优于 scratch」的关键证据，建议重点突出。

---

**文档版本**：v1  
**对应周次**：2026-03-09 至 2026-03-14  
**依据文档**：`docs/PROJECT_OVERVIEW_AND_TRAINING_PLAN.md` 第十节 10.5.1–10.5.4、10.6
