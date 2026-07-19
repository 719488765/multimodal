# 毕业论文实验支撑：全量实验与工程落地主文档

**版本**：v2.0（2026-07-16）  
**适用范围**：仅限本仓库 **`project/`** 子项目（多模态驾驶员情绪分析 + `emotion-agent` 工程配套）。  
**数据口径**：分类指标以各 run 目录下 **`metrics.csv`** 中 `phase=val` 行为主；重要结论建议与 `scripts/recompute_val_metrics.py` 对选定 checkpoint **交叉验证**。

**维护**：重大实验节点（R4 close-out、中文微调、智能体定稿、论文送审前）请更新 **§13 修订记录**，并刷新 §6 / §14 数值表；**调优与工程落地**以 **§12** 为准；**前端推理模型下拉选项**以 **§14** 为准。

**本次 v2.0 增量**：纳入 **SDAVT v3 R4 全队列（55 jobs）**、**M3_M7 中文 BERT 微调 v1/v2（全量）**、**emotion-agent Checkpoint Preset 决策矩阵**，与早期 `logs_rerun` / `logs_accuracy_seq` 形成可写入论文实验章的完整消融链。

---

## 1. 文档目的与阅读路线

本文件将下列材料**按时间线与科学问题**整合为一条可写入学位论文「实验部分」的主线：

| 材料 | 路径（相对仓库根或 `project/`） | 在本文件中的角色 |
|------|-----------------------------------|------------------|
| 顶会论文与方向综述 | `article_guide.md` | §2.1 文献与创新点对齐 |
| 数据集应用策略 | `dataset_application_guide.md` | §3 数据与域偏移讨论 |
| 研究技术路线 | `research_guide.md` | §2.2 方案与模块选型 |
| 工程与训练实现细节 | `project/详细文档.md` | §4 模型与训练实现索引 |
| 混合训练问题分析 | `project/docs/MIXED_DATASET_TRAINING_ANALYSIS.md` | §5.3 异质数据与损失口径 |
| 主线重跑全记录 | `project/docs/EXPERIMENT_RERUN_FULL_RECORD_20260407.md` | §6.1、`logs_rerun`；**v4 起含 R4 + 中文微调附录** |
| 准确率序列指南与 AP4 手册 | `project/docs/EXPERIMENT_ACCURACY_SEQUENCE_GUIDE.md` | §6.2、§9 |
| 准确率序列主记录 | `project/docs/EXPERIMENT_ACCURACY_SEQ_MAIN_RECORD.md` | §6.2 与 AP3 清理说明 |
| SDAVT v3 R4 自动汇总 | `project/docs/SDAVT_V3_R4_EXPERIMENT_RESULTS.md` | §6.4、§7.6–7.8 |
| **SDAVT v3 R4 完整报告（权威长文）** | **`project/docs/R4_FULL_EXPERIMENT_REPORT.md`** | **R4 各组指标、消融、中文微调、Agent Preset §12** |
| 智能体工程方案 | `project/docs/EMOTION_AGENT_ENGINEERING_PLAN.md` | §10、§14 |
| Agent Checkpoint Preset | `emotion-agent/backend/app/core/config.py` | §14 前端下拉选项权威源 |

**本文件不替代**上述专著级文档；论文写作时应在方法节**引用** `详细文档.md` 的模块定义，在实验节以**本文件表格 + 原始 CSV** 为数值锚点。

---

## 2. 文献脉络、研究问题与项目目标

### 2.1 与参考论文的对应关系（摘录整合）

来自 `article_guide.md` / `research_guide.md` 的共性结论：本项目的模型侧重点与近年工作存在明确对应，可在论文「相关工作」中写成**有来源的设计动机**，而非空泛陈述。

- **跨模态融合 + 情感转变**：CFN-ESA（Jiang et al., arXiv:2307.15432）与本项目 **`emotion_shift`** 融合模块在问题设定上高度同构，适合作为**方法参照与对比叙事**。
- **领导–跟随式音视频融合**：ICCV 2021 连续情感识别中的 leader-follower attentive fusion（Zhang et al., arXiv:2107.01175）与本项目 **`leader_follower`** 消融（AP3）对应。
- **多模态高阶相关 / 自监督**：MFMC 等工作（见 `article_guide.md`）可与本项目 **functional correlation / FMC 开关**（若启用）在讨论节关联。
- **情绪–驾驶行为**：CPSOR-GCN 等强调情绪对轨迹/行为的影响，可支撑本课题「**智能体 + 情绪理解**」的应用动机段落。

### 2.2 研究目标（与 `research_guide.md` 一致）

在**智能驾驶人机交互**场景下，对驾驶员 **音视频文** 多模态输入进行情绪识别；技术路线为 **注意力融合 + 预训练（多数据集）+ 可选微调**；数据集组合策略在指南中归纳为「预训练 + 微调」类方案，本项目已落实 **CREMA-D / MELD / CMU-MOSEI** 三源混合预训练与单域上界实验。

### 2.3 本文对「创新点」的诚实界定（支撑开题/中期/答辩用语）

可支撑学位论文的**实证型创新**通常包括（需与导师口径最终对齐）：

1. **系统级**：在统一代码库中完成 **多模态融合 × 域适应 × 多阶段消融矩阵**（`logs_rerun` + `logs_accuracy_seq`），并落地 **emotion-agent** 采集–推理–LLM 编排闭环（见 `EMOTION_AGENT_ENGINEERING_PLAN.md`）。  
2. **方法级**：在混合三数据集、统一 7 类验证协议下，量化 **emotion_shift 相对 standard 的收益**（`logs_rerun`），并在 **R4 单域队列（55 jobs）** 上复现融合/配方/模态消融；MELD 冠军 **M3_M7 F1=0.6957**，另完成 **中文 BERT 微调 v2（F1=0.6114）** 支撑在线 Agent。  
3. **分析级**：对 **ClassBalancedLoss 与 val 总 loss 脱钩**、**Best 与 Last 不一致**、**英文离线 vs 中文部署分轨** 等现象给出基于日志的机制解释（见 `EXPERIMENT_RERUN_FULL_RECORD_20260407.md` v4 与 `MIXED_DATASET_TRAINING_ANALYSIS.md`）。

**不宜过度宣称的表述**：在**未做驾驶舱专有数据集大规模微调**的前提下，不宜将混合验证 Acc 与某一单数据集 SOTA 论文的表格数值直接并列为「超越 SOTA」；更稳妥的写法是 **协议不同、任务为跨域混合分类 + 工程系统**，强调 **可复现消融与智能体集成**。

---

## 3. 数据集与实验协议（整合 `dataset_application_guide.md` 与 `详细文档.md`）

### 3.1 已使用数据源

| 数据集 | 模态 | 类别数（原始） | 在本项目中的角色 |
|--------|------|----------------|------------------|
| CREMA-D | 音/像（及对齐文本若配置） | 6 类 | 混合预训练域之一；**单域实验须注明 6 类**（AP1） |
| MELD | 音/像/文 | 7 类 | 对话式情感，域偏移代表 |
| CMU-MOSEI | 音/像/文 | 7 类 | 真实场景 YouTube，域偏移大 |

### 3.2 域偏移与标注异质性

`MIXED_DATASET_TRAINING_ANALYSIS.md` 已系统归纳：**实验室表演 vs 电视对话 vs 网络视频** 导致的分布差异、类别语义对齐成本、以及 **ClassBalanced** 在 batch 级重算带来的 **验证 loss 与 Acc 脱钩**。这些结论应进入论文「挑战与分析」小节，并解释为何主表除 Acc/F1 外还报告 **`cls_ce_unweighted`**（见 `EXPERIMENT_ACCURACY_SEQUENCE_GUIDE.md` §5.1）。

### 3.3 混合验证协议

三数据集混合预训练时，验证集为 **统一映射后的 7 类** 空间（与 `train.py`、配置中 `emotion_classes: 7` 一致）。**AP1 单数据集上界**与 **AP2 起混合主表**不可混排名，须在论文中用**分表 + 脚注**处理。

---

## 4. 模型与训练实现（索引 `详细文档.md`）

`详细文档.md` 对下列内容有**逐文件级**说明，论文方法节应引用该文档而非在本文件重复贴代码：

- **特征提取**：ResNet50 视频、Wav2Vec2-base 音频、BERT-base 文本等（见 §3.1）。  
- **融合变体**：standard、`emotion_shift`、`leader_follower`、`two_stage`、functional correlation（见 §3.2）。  
- **训练脚本**：`scripts/train.py`（预训练/微调、resume、日志与 TensorBoard 初始化）。  
- **推理脚本**：`scripts/inference.py`（与智能体对接的预处理契约）。  

本文件仅强调与实验记录强相关的**工程事实**：同一 checkpoint 必须与**当时训练所用 yaml** 在 `model.attention.fusion_strategy`、`use_domain_adaptation`、模态开关等字段上**一致**，否则推理权重会错位。

---

## 5. 实验日志目录体系与 TensorBoard 说明

### 5.1 四类主目录（`project/` 下，截至 v2.0）

| 目录 | 状态（截至 2026-07-16） | 说明 |
|------|-------------------------|------|
| `logs/` | 无活跃主表；仅见 `archived/` | 早期管线；**主实验数值不以之为准** |
| `logs_rerun/` | **6 组主线重跑均已满 50 val epoch** | 与 `EXPERIMENT_RERUN_FULL_RECORD_20260407.md` v3/v4 一致 |
| `logs_accuracy_seq/` | **AP0–AP4 准确率序列** | 与 `logs_rerun` **强制隔离** |
| `logs_sdavt_v3_r4/` | **R4 论文轨主日志（55+ jobs + 中文微调）** | TensorBoard：**http://127.0.0.1:6008**（`scripts/start_tensorboard_sdavt_r4.sh`）；归档：`logs_sdavt_v3_r4_archived/` |

另有 `logs_archived/` 存放早期失败重试 run，**不纳入主结果表**。

### 5.2 TensorBoard 图像与曲线：整体分析说明

**重要说明**：本仓库中的 TensorBoard 数据以各 run 子目录下的 **事件文件（`events.out.tfevents.*`）** 与 **`metrics.csv`/`metrics.jsonl`** 并存。作为文本化自动化汇总，**本文件无法替代肉眼查看 TensorBoard 曲线形态**（例如振荡频率、是否过拟合平台、DA 是否在中期突刺后坍塌）。  

基于 **`metrics.csv` 全量统计**，可对「整体形态」作如下**与曲线一致的推断**（供答辩预演与补图清单使用）：

1. **AVT + standard（`logs_rerun`）**：**Best@~27 与 Last@49 严重背离**——中期可达可用 Acc，末期塌陷；TensorBoard 上 `val/accuracy` 应呈**中峰后跌**。论文必须 **Best + Last 双报**（v3 记录已强调）。  
2. **AVT + emotion_shift（`logs_rerun`）**：**Best@~19** 为峰值亮点，末轮 Acc 仍中等偏上，但 **val 总 loss 数量级异常**——TB 上可能出现 **loss 与 Acc 走势不同步**；解释口径见 v3 §6.1。  
3. **AT + DA**：常见 **「峰值优于 noDA、末轮变差」**；DA 扫描（AP4）预期在 TB 上重复类似 **Best vs Last** 张力，故 AP4 定稿亦须双指标。  
4. **`logs_accuracy_seq` AP2**：**M3/M4a** 等出现 **Last 远低于 Best**（末段不稳定），TB 上应重点截 **峰值窗口** 与 **末段 10 epoch** 作对照图。  
5. **AP3**：**leader_text** 末轮 loss 极高、Acc 低，但 **Best Acc 并不差**——TB 上可能呈**大漂移**；写作时建议分开展示 **F1/Acc** 与 **cls_ce_unweighted**。  
6. **R4（`logs_sdavt_v3_r4`）**：单域协议下曲线更「干净」；M3_M7 在 ep≈31 达峰后可早停；中文 v2 在 **ep5** 达 Best F1=0.6114，**ep10 early-stop**。TB 端口 **6008**。

**建议你在论文定稿前完成的 TB 工作**（本文件无法代劳）：对每一主 run 导出 **`val/accuracy`、`val/f1`、`val/cls_ce_unweighted`、`train/loss_*`** 四宫格截图，按 AP0→AP4→R4→中文微调 顺序放入 `project/docs/figures/tensorboard/`（自建目录），并在 **§13 修订记录** 中登记导出日期。

---

## 6. 全量数值汇总（来自 `metrics.csv`，`phase=val`）

### 6.1 主线重跑 `logs_rerun/`（六组，与 v3 记录一致）

| run | n_val | epoch | Last Acc | Last F1 | Best Acc @ep | Best F1 @ep |
|-----|-------|-------|----------|---------|--------------|-------------|
| `RERUN_AT_pretrain_3datasets_noDA_20260401_021657` | 50 | 0–49 | 0.2549 | 0.2282 | 0.3224 @19 | 0.2538 @27 |
| `RERUN_AT_pretrain_3datasets_DA_20260403_134709` | 50 | 0–49 | 0.1820 | 0.1343 | 0.3526 @24 | 0.3128 @24 |
| `RERUN_VT_pretrain_3datasets_noDA_20260403_134709` | 50 | 0–49 | 0.3736 | 0.3406 | 0.3796 @45 | 0.3474 @45 |
| `RERUN_AVT_pretrain_3datasets_DA_20260403_134709` | 50 | 0–49 | 0.2765 | 0.2260 | 0.3537 @42 | 0.2709 @42 |
| `RERUN_AVT_pretrain_3datasets_noDA_20260407_193400`（standard） | 50 | 0–49 | 0.1701 | 0.1232 | 0.3726 @27 | 0.3287 @27 |
| `RERUN_AVT_pretrain_3datasets_noDA_emotion_shift_20260407_193400` | 50 | 0–49 | 0.3747 | 0.2998 | **0.4455 @19** | **0.4036 @19** |

**与 v3 文档的交叉验证**：上表与 `EXPERIMENT_RERUN_FULL_RECORD_20260407.md` §4 一致，可作为论文「主线重跑」主表数据源。

### 6.2 准确率优化序列 `logs_accuracy_seq/`（AP0–AP4）

#### AP0（noDA standard 对照）

| run | n_val | Last Acc | Best Acc @ep | Best F1 @ep |
|-----|-------|----------|--------------|-------------|
| `AP0_AVT_pretrain_3datasets_noDA_standard_full50_s3407_20260413_202830` | 50 | 0.3164 | 0.3245 @1 | 0.2152 @2 |

#### AP1（单域上界）

| run | n_val | Last Acc | Best Acc @ep | Best F1 @ep |
|-----|-------|----------|--------------|-------------|
| `AP1_VT_pretrain_crema_only_s3407_20260420_202232` | 50 | 0.1747 | 0.1788 @0 | 0.0542 @0 |
| `AP1_VT_pretrain_meld_only_s3407_20260420_202232` | 50 | 0.4233 | 0.4233 @0 | 0.2518 @0 |
| `AP1_VT_pretrain_mosei_only_s3407_20260420_203623` | 50 | 0.7162 | 0.7162 @0 | 0.5978 @0 |
| `AP1_AVT_ES_pretrain_crema_only_s3407_20260420_202232` | 50 | 0.2997 | 0.3454 @42 | 0.3039 @42 |
| `AP1_AVT_ES_pretrain_meld_only_s3407_20260420_202232` | 50 | 0.5740 | 0.5803 @36 | 0.5395 @36 |
| `AP1_AVT_ES_pretrain_mosei_only_s3407_20260420_203623` | 50 | 0.6986 | 0.7162 @0 | 0.6119 @2 |

#### AP2（三混合 + emotion_shift 配方）

| run | n_val | Last Acc | Best Acc @ep | Best F1 @ep |
|-----|-------|----------|--------------|-------------|
| `AP2_ES_3ds_baseline_s3407_20260422_210615` | 50 | 0.4620 | 0.5909 @9 | 0.5320 @12 |
| `AP2_M1_ES_3ds_effbatch8_s3407_20260422_210615` | 50 | 0.5547 | **0.6068 @6** | 0.5623 @7 |
| `AP2_M2_ES_3ds_lr5e5_s3407_20260422_210615` | 50 | 0.5713 | 0.6054 @2 | 0.5587 @1 |
| `AP2_M3_ES_3ds_uniform_s3407_20260422_210615` | 50 | 0.2146 | 0.5931 @2 | 0.5521 @3 |
| `AP2_M4a_ES_3ds_plain_ce_s3407_20260422_210615` | 50 | 0.1797 | 0.5799 @0 | 0.5365 @4 |
| `AP2_M4b_ES_3ds_focal_s3407_20260422_210615` | 50 | 0.3935 | 0.5974 @3 | 0.5523 @3 |

#### AP3（融合消融，失败 run 已从 TB 目录删除，见 `EXPERIMENT_ACCURACY_SEQ_MAIN_RECORD.md`）

| run | n_val | Last Acc | Best Acc @ep | Best F1 @ep | 备注 |
|-----|-------|----------|--------------|-------------|------|
| `AP3_fusion_standard_3ds_s3407_20260501_110235` | 50 | 0.5592 | 0.5630 @35 | 0.5175 @31 | 末段相对稳定 |
| `AP3_fusion_leader_text_3ds_s3407_20260501_110721` | 50 | 0.2844 | 0.5976 @7 | 0.5415 @6 | 末段劣化大，解读需分项 loss |
| `AP3_fusion_leader_audio_3ds_s3407_20260501_110721` | 50 | 0.5501 | 0.5670 @43 | 0.5247 @43 | |
| `AP3_fusion_two_stage_3ds_s3407_20260501_110720` | 50 | 0.5173 | 0.5173 @0 | 0.3528 @0 | **已满 50 epoch**（2026-05-16 核对）；val 指标自 ep0 平台恒定 |

#### AP4（域适应扫描）

| run | n_val | 说明 |
|-----|-------|------|
| `AP4_AVT_pretrain_3datasets_DA_20260508_200639` | 0（快照时） | DA 默认配置已启动；**待产生 val 后回填** §6.2 表体 |
| 其余 6 个 yaml | — | 按 `EXPERIMENT_ACCURACY_SEQUENCE_GUIDE.md` §4.4 分会话执行 |

### 6.3 `outputs_accuracy_seq/`

当前存在 `accuracy_seq_results_summary.csv` / `.md` 等导出物，其内容**可能早于**完整 50 epoch 训练（尤其 AP2 早期快照）。**论文制表以 §6.2 上表及原始 `metrics.csv` 为准**；可在 AP4 结束后统一重跑汇总脚本并修订本节说明。

### 6.4 SDAVT v3 R4 全队列（`logs_sdavt_v3_r4/`，2026-06～07）

**权威自动表**：[`SDAVT_V3_R4_EXPERIMENT_RESULTS.md`](SDAVT_V3_R4_EXPERIMENT_RESULTS.md)（`scripts/build_sdavt_r4_report.py`，快照 2026-07-09，**done=55**）。  
**权重根目录**：`checkpoints_sdavt_v3_r4/`。  
**协议**：单域训练/验证（CREMA / MELD / MOSEI），与混合 AP2 **不可混排名**。

#### 6.4.1 Phase 冠军与代表值（Best val F1）

| Phase | 代表 Job | Dataset | Best F1 | Best Acc | 日志 / Agent 映射 |
|-------|----------|---------|---------|----------|-------------------|
| p1_baseline | R4_B_O0 | mosei | 0.6792 @12 | 0.7269 | `SDAVT_R4_R4_B_O0_*` |
| p1_baseline | R4_B_C0 | crema | 0.5889 @19 | 0.5874 | `SDAVT_R4_R4_B_C0_*` |
| p1_baseline | R4_B_M1 | meld | 0.5680 @3 | 0.5966 | `SDAVT_R4_R4_B_M1_*` |
| p2_fusion | F_O_ES | mosei | 0.6792 @12 | 0.7269 | → preset `sdavt_mosei_r4` |
| p2_fusion | F_M_ES | meld | 0.6109 @3 | 0.6245 | `SDAVT_R4_F_M_ES` |
| p2_fusion | F_C_ES | crema | 0.5786 @28 | 0.5860 | `SDAVT_R4_F_C_ES` |
| p3_c3 | C3_C2_w2v_large | crema | 0.5629 @31 | 0.5672 | 队列配方；Agent 现用 C4_C3 |
| **p3_m3** | **M3_M7_combo** | **meld** | **0.6957 @31** | **0.7121** | → preset **`sdavt_meld_v3_r4`** |
| p3_m3 | M3_M1_roberta | meld | 0.6823 @17 | 0.6968 | 仅换 RoBERTa |
| p4_modal | R4_A_O_T | mosei | 0.7087 @15 | 0.7483 | 文本主导 |
| p4_modal | R4_A_M_AVT | meld | 0.6818 @5 | 0.6958 | 三模态对齐 M3 |

#### 6.4.2 中文微调轨（接在 M3_M7 之后）

| Run / Preset | 文本骨干 | 训练数据 | Best val F1 | Best Acc | Checkpoint |
|--------------|----------|----------|-------------|----------|------------|
| M3_M7_combo（英文主轨） | roberta-base | MELD 英文字幕 | **0.6957** | **0.7121** | `checkpoints_sdavt_v3_r4/SDAVT_R4_M3_M7_combo/checkpoint_pretrain_best_f1.pth` |
| chinese_agent **v1** | bert-base-chinese | MELD（英文字幕 + 中文词表） | **0.6010** @ep9 | 0.6273 | → **`sdavt_meld_zh_agent`** |
| chinese_agent **v2 全量** | bert-base-chinese | MELD + 500×`*_zh.txt` + 97 agent_capture | **0.6114** @ep5 | **0.6363** | → **`sdavt_meld_zh_agent_v2`**（默认部署） |

日志：`outputs_sdavt_v3_r4/status/m3m7_zh_finetune.log`、`m3m7_zh_v2_full_finetune.log`；metrics：`logs_sdavt_v3_r4/SDAVT_R4_M3_M7_chinese_agent{,_v2}/metrics.csv`。

**分析**：换中文 BERT 后离线 F1 相对英文冠军约 **−9.5 pt**（监督仍偏英文）；v2 中文增强后 **+1.0 pt**；在线中文 E2E（2026-07-16）`我很难过` → sad conf≈**0.534**。论文须分表「离线英文 MELD」与「在线中文 Agent」，禁止混排名。

---

## 7. 消融对比与实验评价（压缩进论文「结果与讨论」的骨架）

### 7.1 模态（AT → VT → AVT）

- **VT 相对 AT**：末轮 Acc 提升约 **+0.12**（见 `EXPERIMENT_RERUN_FULL_RECORD_20260407.md` §5.2 差分表），是本轮最清晰的模态增益信号。  
- **AVT + DA 相对 VT**：在固定超参下**未超越** VT 的末轮/峰值（同表 §5.3），适合作为「复杂度–收益」讨论点，而非简单否定三模态价值。

### 7.2 融合（standard vs emotion_shift，主线）

- **emotion_shift 相对 standard**：**Best Acc +0.073～0.09 量级、末轮 Acc 大幅提升**（v3 §5.4），与 CFN-ESA 类叙事一致，可作为**核心消融结论**。  
- **standard 末段塌陷**：必须在论文中 **Best@27 + Last@49** 并列，避免审稿人质疑「挑轮次」。

### 7.3 配方（AP2）

- **有效 batch 扩大（M1）** 与 **lr 降低（M2）** 将混合 **Best Acc** 推至 **~0.61 / ~0.61**，高于主线 `logs_rerun` 中 emotion_shift 峰值 **~0.45**（注意：**二者 yaml 协议、配方、日志字段代际不完全相同**，并列时须在表注写清「不同实验批次」）。  
- **M3 uniform**、**M4a 纯 CE** 末轮表现差，提示 **采样与损失形态** 对稳定性极敏感。

### 7.4 融合结构（AP3）

- **standard** 与 **leader_audio** 的 **Last** 较优；**leader_text** **Best 高而 Last 差**，提示文本主导在该三域混合下可能过拟合或损失加权异常。  
- **two_stage** 已满 50 epoch，但 **Acc/F1 全程平台化**（≈0.52/0.35），论文中宜作「融合形态对照」并说明早停平台，不宜强调末段提升。

### 7.5 域适应（AP4）

- AP4 w005 等已写入 Agent preset `ap4_w005`（混合 Best F1≈0.528）；相对 AP2 M1（≈0.56）为 **DA 对照**而非默认部署。

### 7.6 R4：融合策略（p2，同域内）

| Dataset | emotion_shift (ES) | standard (STD) | Δ F1 (ES−STD) | 结论 |
|---------|-------------------|----------------|---------------|------|
| MELD | F_M_ES **0.6109** | F_M_STD 0.4447 | **+0.166** | ES 显著优于 STD |
| CREMA | F_C_ES **0.5786** | F_C_STD 0.2405 | **+0.338** | ES 必需；STD 近崩溃 |
| MOSEI | F_O_ES **0.6792** | F_O_STD 0.5988 | **+0.080** | ES 仍优，增益小于 MELD |

与早期 `logs_rerun`「AVT emotion_shift ≫ standard」结论一致，且在单域协议下更干净。

### 7.7 R4：MELD 配方消融（p3_m3）

| Job | 改动 | Best F1 | 相对 M0 Δ | 解读 |
|-----|------|---------|-----------|------|
| M3_M0_baseline | 基线 ES | 0.6080 | — | 锚点 |
| M3_M1_roberta | 文本→RoBERTa | **0.6823** | **+0.074** | 文本骨干关键增益 |
| M3_M2_w2v_large | 音频→wav2vec-large | 0.5572 | −0.051 | 单纯换 large 未增益 |
| M3_M3_uniform | 均匀采样 | 0.6105 | +0.002 | 边际 |
| M3_M4_focal | focal | 0.6079 | ≈0 | 边际 |
| M3_M5_context | context window | 0.5725 | −0.036 | 未改善 |
| M3_M6_moddrop | modality dropout | 0.6079 | ≈0 | 边际 |
| **M3_M7_combo** | RoBERTa+配方组合 | **0.6957** | **+0.088** | **论文/英文 Agent 冠军** |

### 7.8 R4：模态消融（p4_modal，要点）

| Dataset | 强组合 | Best F1 | 弱/崩溃组合 | 含义 |
|---------|--------|---------|-------------|------|
| MELD | T / AT / AVT / VT | ≈0.67–0.68 | V alone collapse | 文本关键；纯视觉不足 |
| MOSEI | T / VT / AVT / AT | ≈0.69–0.71 | — | 文本主导最强 |
| CREMA | V / AV / AVT | ≈0.33–0.35 | T / A / AT collapse | 表演语料文本极弱 |

→ Agent 中文场景用 **leader_audio + 可选 skip_text**（`config_agent_deploy.yaml`）与 CREMA/中文 ASR 域差一致；英文对话用 **文本强** 的 M3_M7。

### 7.9 中文微调消融（v1 / v2）

| 对比 | Δ val F1 | 结论 |
|------|----------|------|
| v1 中文 BERT vs M3_M7 英文 RoBERTa | **−0.095** | 词表适配换骨干有代价（监督仍英文） |
| v2 全量 vs v1 | **+0.010** | 中文伪标签+采集注入有效 |
| v2 vs M3_M7（协议不同） | 勿直接比 | 部署选中文轨；论文主表选 M3_M7 |

**不建议**：对 MOSEI/CREMA 做中文 BERT 微调（英文语料无中文监督，ROI 低）——详见 §14。

---

## 8. 与参考复现论文的数值关系：能否支撑学位论文？

**结论（可写进开题/中期答辩的谨慎表述）**：

- **支撑点**：你已具备 **可溯源的多阶段实验矩阵**（重跑 + accuracy_seq）、**与顶会方法名词对齐的模块实现**、以及 **智能体工程方案**。在**工学/计算机专硕**范式下，这足以构成「**算法 + 系统 + 消融**」型论文主体。  
- **风险点**：若评审期望 **在驾驶专有数据上达到某篇单数据集 SOTA 的同样 Acc**，当前混合验证数值**未必**满足；应提前将贡献定位为 **跨域混合训练规律 + 消融 + 智能体闭环**，并规划 **AP5 微调**（指南阶段 5）或自有采集数据的小规模微调作为「展望/后续工作」。

**是否足以「开始撰写毕业论文实验章节」**：**可以**。建议结构为：数据与协议（§3）→ 实现要点引用 `详细文档.md`（§4）→ 主线六组结果（§6.1）→ 准确率序列 AP1–AP3（§6.2）→ **R4 单域消融主表（§6.4）** → **中文微调与 Agent（§6.4.2、§14）** → 讨论与局限（§7–§8）→ 智能体系统（§10）。

---

## 9. 当前进程结论（截至 v2.0，2026-07-16）

1. **主线重跑已闭环**：`logs_rerun` 六组满 50 epoch；**emotion_shift** 为 AVT noDA 设置下最强峰值（Best Acc≈0.445）。  
2. **准确率序列**：AP0–AP3 与 AP4 部分已跑；混合验证峰值约在 **AP2 M1/M2 Best Acc≈0.61**（与 rerun 批次隔离）。  
3. **R4 单域队列已闭环（55 jobs）**：MELD 冠军 **M3_M7_combo Best F1=0.6957**；MOSEI **F_O_ES≈0.679**；CREMA close-out **C4_C3 Acc≈0.605**（Agent preset）。权威表见 `SDAVT_V3_R4_EXPERIMENT_RESULTS.md` / `R4_FULL_EXPERIMENT_REPORT.md`。  
4. **中文微调轨已定稿**：**v2 全量** Best val F1=**0.6114** @ep5（优于 v1 的 0.6010）；默认 Agent preset = **`sdavt_meld_zh_agent_v2`**。  
5. **工程侧**：`emotion-agent` 已绑定真实权重；支持 **语言自动路由**（zh→v2，en→`sdavt_meld_v3_r4`）；前端下拉选项以 **§14** 为准。

---

## 10. 智能体服务：checkpoint 选取与使用方案

### 10.1 推荐策略（分场景，v2.0 更新）

| 场景 | 推荐 preset / checkpoint | 理由 |
|------|--------------------------|------|
| **A. 论文英文 MELD 主表 / 英文对话演示** | preset **`sdavt_meld_v3_r4`**（M3_M7 F1=0.6957） | R4 单域冠军，与 §6.4 / §7.7 严格对齐 |
| **B. 中文在线 Agent（默认部署）** | preset **`sdavt_meld_zh_agent_v2`**（F1=0.6114） | 中文 BERT + 中文增强；`.env` 已设为默认 |
| **C. 中文消融对照** | preset **`sdavt_meld_zh_agent`**（v1，F1=0.6010） | 证明 v2 增量；勿作默认 |
| **D. 论文叙事与早期主线对齐** | `checkpoints_rerun/...emotion_shift.../checkpoint_pretrain_best_f1.pth` | 与 `logs_rerun` 六组图对齐 |
| **E. 混合三域峰值演示** | preset **`ap2_m1`** | 混合 Best Acc≈0.61；与 R4 **不可混排名** |
| **F. 仅实验（勿默认）** | `sdavt_mosei_r4` / `sdavt_crema_r4` | 单域对照；中文场景不适用 |

**默认建议**：论文实验章主表以 **A（M3_M7）+ R4 消融** 为主；系统实现章声明默认权重为 **B（中文 v2）**；英文用户走 **A**（自动路由）。完整 UI 列表见 **§14**。

### 10.2 与 `emotion-agent` 对接的操作要点

配置文件：`emotion-agent/backend/.env`

```env
MODEL_PROVIDER=current
MODEL_CHECKPOINT_PRESET=sdavt_meld_zh_agent_v2
PROJECT_ROOT=/home/lizhichun_24/sda1/code/multimodal/project
MODEL_DEVICE=cuda
```

**必须满足**：

1. Preset 对应的 `train_config` 与 `checkpoint` 成对（见 `CHECKPOINT_PRESETS`）；勿混用不同 run 的 yaml/权重。  
2. 显式传 `checkpoint_preset` 时覆盖自动路由；未指定且开启 auto 时，中文→v2、英文→`sdavt_meld_v3_r4`。  
3. 离线冒烟：`scripts/inference.py` 对同一对 `(config, checkpoint)` 验证后再接 HTTP。

### 10.3 与 `scripts/inference.py` 的一致性

推理侧预处理（帧数、采样率、文本分词器）必须与 yaml 中 `data.*` / 文本骨干一致；中文 preset 使用 **bert-base-chinese**，英文 M3_M7 使用 **RoBERTa**。

---

## 11. 后续优化摘要（指向 §12）

**是否还需要安排后续实验？** **需要。** 混合三数据集验证 Acc 与单域上界差距大，且智能体在线分布与预训练分布不一致；**仅靠当前预训练权重难以同时满足「论文章节完整性」与「上线主观效果」**。具体调优路线、分阶段命令与工程落地清单见 **§12**（本文件为毕业论文实验与工程部分的**执行蓝本**，建议在定稿前至少完成 §12.3 中的 **G0–G2 + G4**）。

---

## 12. 后续调优与工程落地专项方案（提准确率 + 智能体定稿）

本节回答三件事：**（1）为何还要做实验；（2）按什么顺序做能同时抬高指标与降低工程风险；（3）每一步的具体操作**（命令级）。默认工作目录为 **`project/`** 根（含 `scripts/`、`config/`、`data/`）。

### 12.1 结论：为何需要「继续实验 + 工程补强」

| 现象 | 含义 | 对策方向 |
|------|------|----------|
| 混合验证 Acc 明显低于 AP1 单域 | 域偏移 + 类别噪声 + 小有效 batch | **先完成 AP4** 扫域适应；再 **目标域/单域微调**（AP5）；工程上 **滑窗投票 + 置信度门控** |
| `logs_rerun` 与 `logs_accuracy_seq` 最优配置不统一 | 配方/epoch/日志字段代际不同 | **选定一条「主发布线」yaml**，后续微调与智能体只跟这一条 |
| `emotion-agent` 未绑定真实权重 | 演示仍为 mock 或未加载 | **定版 deploy 配置 + checkpoint + 冒烟脚本**（§12.5 G4） |
| 末轮与 Best 差距大 | 训练后期校准/域损失干扰 | **早停导出 best_f1**；论文与上线均以 **best_f1** 为默认服务权重 |

**工程向核心原则**：先把 **「可复现的一条训练配置 + 一个 best_f1 权重」** 冻结为 **Deploy Baseline**，再在其上做 **小步微调** 与 **在线策略**；避免「同时改模型、改前后端、改数据」导致无法归因。

---

### 12.2 总体路线图（阶段编号 G0–G6）

| 阶段 | 名称 | 目的 | 依赖 GPU |
|------|------|------|----------|
| **G0** | 预训练闭环收尾 | AP4 七组（分会话）、AP3 `two_stage` 满 50 epoch | 有 |
| **G1** | 主发布线选型 | 在数值与稳定性间选定 **唯一**「智能体 + 论文共用」backbone ckpt | 无 |
| **G2** | Deploy 配置镜像 | 从选定 yaml **复制**出 `config/config_agent_deploy.yaml`，仅改 `paths`/`experiment.name` 便于追溯 | 无 |
| **G3** | 目标域微调（AP5） | 在 **CREMA / MELD / MOSEI 之一** 上 `--mode finetune --resume`，抬高该域或 6/7 类上线口径 | 有 |
| **G4** | emotion-agent 绑定 | `.env` 指向 `MODEL_CONFIG_PATH` + `MODEL_CHECKPOINT_PATH`；后端加载与 `inference.py` 契约一致 | 有（推理测试） |
| **G5** | 在线工程增强 | 滑窗、多帧投票、VAD、ASR 低置信度降级（见 `EMOTION_AGENT_ENGINEERING_PLAN.md`） | 弱 |
| **G6** | 论文与验收材料 | `recompute_val_metrics.py`、延迟与失败率日志、TB 截图归档 | 无 |

**推荐默认「主发布线」**（在 G1 最终确认前可按需替换）：

- **数值优先（混合验证 Acc 最高）**：`checkpoints_accuracy_seq/AP2_M1_ES_3ds_effbatch8_s3407_20260422_210615/checkpoint_pretrain_best_f1.pth`，配置母本为 `config/rerun/accuracy_plan/ap2_M1_effbatch8_ES_3ds_s3407.yaml`。  
- **叙事优先（与主线重跑 emotion_shift 对齐）**：`checkpoints_rerun/RERUN_AVT_pretrain_3datasets_noDA_emotion_shift_20260407_193400/checkpoint_pretrain_best_f1.pth`，母本为对应 `config/rerun/` 下该次训练 yaml（以你实际训练时保存的 yaml 为准，可从 `docs/EXPERIMENT_RERUN_FULL_RECORD_20260407.md` 与 `outputs_rerun` 制表元数据反查）。

下文命令中 **`$CKPT_MAIN`**、`$YAML_MAIN`** 请你在 G1 结束后替换为最终路径。

---

### 12.3 G0：预训练闭环（算法侧，优先）

**目标**：在不改智能体代码的前提下，先把「该跑完的训练」跑完，避免论文里 AP4/AP3 缺口。

1. **AP4**：按 `docs/EXPERIMENT_ACCURACY_SEQUENCE_GUIDE.md` **§4.4** 分会话启动剩余 6 个配置；**禁止**一键七路并行。  
2. **AP3 two_stage**：若 `logs_accuracy_seq/AP3_fusion_two_stage_3ds_s3407_20260501_110720` 未满 50 epoch，使用 `--resume` 指向同目录下 `checkpoint_pretrain_best.pth` 或 `best_f1`；若已实现环境变量 **`MULTIMODAL_LOG_RUN_DIR_NAME`**（见 `utils/helpers.py`），可复用原日志目录名续写 TB。  
3. **验收**：每个 run 的 `metrics.csv` 中 `phase=val` 行数应等于 `num_epochs`（或你主动早停时的 epoch+1）。

---

### 12.4 G1：主发布线选型（决策表）

在 **G0 完成**后，填写下表并**只选一行**作为 `CKPT_MAIN`（可另存为 `CKPT_PAPER` 若与演示不同）。

| 候选 | checkpoint 路径（示例） | 混合 val Best Acc（约） | Last 稳定性 | 智能体适配难度 |
|------|-------------------------|-------------------------|---------------|----------------|
| AP2 M1 | `.../AP2_M1_ES_3ds_effbatch8_s3407_20260422_210615/checkpoint_pretrain_best_f1.pth` | ~0.61 | 中 | 中：需 AVT+emotion_shift 与 yaml 一致 |
| AP2 M2 | `.../AP2_M2_ES_3ds_lr5e5_s3407_20260422_210615/checkpoint_pretrain_best_f1.pth` | ~0.61 | 中 | 同上 |
| AP3 standard | `.../AP3_fusion_standard_3ds_s3407_20260501_110235/checkpoint_pretrain_best_f1.pth` | ~0.56 | **较好** | 中：fusion 为 standard |
| Rerun ES | `.../RERUN_AVT_..._emotion_shift_20260407_193400/checkpoint_pretrain_best_f1.pth` | ~0.45 | 中 | 中 |

**建议**：**对外演示与「上线主观准确率」优先** → 选 **AP2 M1 best_f1**；**论文与主线重跑图表严格一致** → 选 **Rerun emotion_shift**；二者也可 **论文用 B、演示用 A**，但须在论文「系统实现」中**各写一行路径**。

---

### 12.5 G2：生成 `config_agent_deploy.yaml`（配置镜像）

**目的**：避免直接改 `config/config.yaml` 导致实验不可复现；智能体只读 deploy 专用配置。

```bash
cd /mnt/sda1/lizhichun_24/code/multimodal/project   # 按你的实际路径修改

# 以 AP2 M1 为例：将母本 yaml 复制为 deploy 配置
cp config/rerun/accuracy_plan/ap2_M1_effbatch8_ES_3ds_s3407.yaml config/config_agent_deploy.yaml
```

用编辑器打开 **`config/config_agent_deploy.yaml`**，**至少**确认或修改：

1. **`paths.log_dir` / `paths.checkpoint_dir` / `paths.output_dir`**：若仍希望微调写入隔离桶，可改为 `logs_accuracy_seq/`、`checkpoints_accuracy_seq/`、`outputs_accuracy_seq/`（与 `EXPERIMENT_ACCURACY_SEQUENCE_GUIDE.md` 第六节一致）。  
2. **`experiment.name`**：改为可识别字符串，例如 `AGENT_DEPLOY_AP2M1`，便于 TensorBoard 区分。  
3. **`model.*` 与训练 ckpt 完全一致**（模态开关、`fusion_strategy`、DA 开关等）。

---

### 12.6 G3：微调（AP5，提高「目标域」或上线口径准确率）

**动机**：混合预训练优化的是 **三域联合验证**；智能体若主要面向 **中文语音 + 摄像头** 或 **实验室采集的类 CREMA 片段**，应用 **单数据集微调** 往往比继续盲调混合预训练 **ROI 更高**。

**操作概要**（以 CREMA 6 类微调为例；**若智能体必须是 7 类**，则改用 MELD/MOSEI 单域或保留 7 类 head 的 yaml，需自行复制模板并统一 `emotion_classes`）：

1. **复制模板**：以 `config/config_crema_finetune_from_pretrain.yaml` 为参考，在 `config/rerun/accuracy_plan/` 下新建 `ap5_crema_finetune_from_AP2M1_accuracy_seq.yaml`（文件名自定）。  
2. **修改要点**：  
   - `paths.*` → `logs_accuracy_seq/`、`checkpoints_accuracy_seq/`；  
   - `training.finetune.datasets: ["crema"]`；  
   - `model.output.emotion_classes: 6`；  
   - **模态与融合**：若预训练为 **AVT + emotion_shift**，微调 yaml 必须与之一致（**不要**沿用仓库默认 AT-only 的 crema 模板而不改模态）。推荐做法：**以 `ap2_M1_effbatch8_ES_3ds_s3407.yaml` 全文为底**，只改 `training` 中 `mode` 相关段、`finetune` 段、`paths`、`experiment.name`。  
3. **启动训练**（单卡 tmux 示例）：

```bash
cd /mnt/sda1/lizhichun_24/code/multimodal/project
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate myenv310
export CUDA_VISIBLE_DEVICES=0

python3 scripts/train.py \
  --config config/rerun/accuracy_plan/ap5_crema_finetune_from_AP2M1_accuracy_seq.yaml \
  --mode finetune \
  --resume /mnt/sda1/lizhichun_24/code/multimodal/project/checkpoints_accuracy_seq/AP2_M1_ES_3ds_effbatch8_s3407_20260422_210615/checkpoint_pretrain_best_f1.pth
```

4. **`train.py` 行为**：`--mode finetune` 下对已变更类别数的分类头会 **部分跳过加载**（见脚本内注释）；属预期。  
5. **验收**：微调 run 的 `metrics.csv` 中 val **Best F1** 相对微调前 **在 CREMA 验证子集上** 应有可见提升，再将该 **微调后 `checkpoint_pretrain_best_f1.pth`（或 finetune 命名规则下的 best）** 定为 **`CKPT_AGENT`**。

---

### 12.7 G4：emotion-agent 工程落地（绑定模型）— **已实现 2026-05-23**

**配置文件**：`emotion-agent/backend/.env`

```env
MODEL_PROVIDER=current
MODEL_CHECKPOINT_PRESET=ap2_m1
PROJECT_ROOT=/home/lizhichun_24/sda1/code/multimodal/project
MODEL_DEVICE=cuda
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

- `ap2_m1`：自动绑定 AP2 M1 `checkpoint_pretrain_best_f1.pth`（混合 val Best F1≈0.562）——**历史演示 preset**  
- **当前默认（2026-07）**：`MODEL_CHECKPOINT_PRESET=sdavt_meld_zh_agent_v2`；英文场景用 `sdavt_meld_v3_r4`；完整列表见 **§14**  
- `ap4_w005`：切换 preset 即绑定 AP4 w005 checkpoint（F1≈0.528）  
- 实现：`utils/emotion_inference_service.py` + `CurrentProjectAdapter`；详见 `emotion-agent/backend/README_DEPLOY.md`

**步骤清单**：

1. ~~适配层~~ 已完成。  
2. **离线冒烟**（必须在接 UI 前执行）：

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project
conda activate myenv310
python3 scripts/inference.py \
  --config config/rerun/accuracy_plan/ap2_M1_effbatch8_ES_3ds_s3407.yaml \
  --model_path checkpoints_accuracy_seq/AP2_M1_ES_3ds_effbatch8_s3407_20260422_210615/checkpoint_pretrain_best_f1.pth \
  --video <测试视频路径> \
  --audio <测试音频路径> \
  --text "测试文本"
```

3. 启动 **ASR → backend → frontend**（端口与 `docker`/tmux 约定见工程方案文档），用固定测试用例回归 **延迟、错误率、情绪标签分布**。  
4. **类别映射**：若微调为 6 类而 UI 为 7 类，必须在后端做 **显式映射表** 或在微调中保留 7 类 head；**禁止**静默错位。

---

### 12.8 G5：在线侧「不涨参」提效果（工程优先）

下列项**不增加训练成本**，常能改善主观准确率：

1. **时间滑窗 + 多数投票**：3 s 窗、1 s 步长（见 `EMOTION_AGENT_ENGINEERING_PLAN.md` §4.2）；对输出标签做 **最近 K 窗众数**。  
2. **低置信度拒识**：若后端输出 softmax 最大概率低于阈值（如 0.45），返回「不确定」而非强行分类。  
3. **ASR 置信度加权**：Whisper 置信度低时降低文本分支对融合的贡献（规则或门控，不必重训）。  
4. **VAD 去静音**：减少无效窗送入模型。

---

### 12.9 G6：论文与验收材料

1. 对 **`CKPT_AGENT`** 运行 `scripts/recompute_val_metrics.py`（参数以脚本 `-h` 为准），与 `metrics.csv` **Best 行**对照。  
2. 导出 TensorBoard **val/accuracy、val/f1、val/cls_ce_unweighted`** 曲线入 `project/docs/figures/`。  
3. 记录 **单请求推理耗时（P50/P95）** 与 **GPU 显存峰值**，写入论文「系统评估」小节。

---

### 12.10 若微调后混合 Acc 仍不满意：第二轮算法实验（可选 G+）

在 **G0–G3 已完成**后再启动，避免与 AP4 抢显存：

| 编号 | 内容 | 说明 |
|------|------|------|
| G+.1 | 在 **AP2 M1** 配方上 **多种子**（如 3407、42、114）各跑满程 | 论文可报告均值±方差 |
| G+.2 | **早停 + 固定 best_f1** | 减少末段塌陷对上线的影响 |
| G+.3 | **domain_loss 余弦调度** | 缓解 DA 末轮劣化（需在 `train.py` 或配置中扩展，属代码级小改） |
| G+.4 | **自采/合规驾驶舱小数据半监督** | 强应用向，需数据合规审查 |

---

### 12.11 本节小结（执行顺序口诀）

**先闭环（G0）→ 选型（G1）→ 镜像配置（G2）→ 微调（G3）→ 绑 agent（G4）→ 在线策略（G5）→ 写材料（G6）**；算法第二轮（G+）仅在工程验收后仍有指标压力时启用。

---

## 13. 修订记录

| 日期 | 版本 | 摘要 |
|------|------|------|
| 2026-05-08 | v1.0 | 首版：整合文献/指南/`详细文档`、三类日志数值表、TB 分析说明、checkpoint 与 emotion-agent 方案、论文可行性评估与后续实验路线。 |
| 2026-05-08 | v1.1 | 新增 **§12 后续调优与工程落地专项方案**（G0–G6、微调与 agent 操作步骤）；§11 改为摘要并指向 §12；原「修订记录」顺延为 §13。 |
| 2026-07-16 | **v2.0** | 纳入 **R4（55 jobs）§6.4**、**中文微调 v1/v2 §6.4.2/§7.9**、消融扩展 **§7.6–7.9**；重写 **§9–§10**；新增 **§14 Agent 前端模型选择矩阵**；与 `EXPERIMENT_RERUN_FULL_RECORD` v4、`SDAVT_V3_R4_EXPERIMENT_RESULTS.md` 对齐。 |

---

## 14. emotion-agent 前端推理模型选择矩阵（Preset 权威参考）

> **代码权威源**：`emotion-agent/backend/app/core/config.py` 中 `CHECKPOINT_PRESETS` + `PRESET_METADATA`。  
> 前端 `/model/status` 透传 metadata；下拉框应按 **group / language / recommended** 分组展示。  
> **论文制表**与 **UI 文案**共用本表，避免两套数字。

### 14.1 完整 Preset 一览（按推荐优先级）

| Preset ID | UI 分组建议 | 语言 | Best val F1 / Acc | 训练协议 | 推荐 | 实验/默认 | 选用场景 |
|-----------|-------------|------|-------------------|----------|------|-----------|----------|
| **`sdavt_meld_zh_agent_v2`** | 中文部署 | zh | **0.6114 / 0.6363** | MELD 中文 BERT 全量微调（自 v1） | ✓ | **默认部署** | 中文 ASR/Agent 在线 |
| **`sdavt_meld_v3_r4`** | 英文/论文 | en | **0.6957 / 0.7121** | R4 M3_M7_combo | ✓ | 英文推荐 | 英文对话、论文主表 |
| `sdavt_meld_zh_agent` | 中文对照 | zh | 0.6010 / 0.6273 | M3_M7→中文 BERT v1 | — | 消融对照 | 相对 v2 −1.0 pt |
| `meld_only` | 历史对照 | en | ≈0.54 | AP1 MELD 单域 | — | 对照 | 早期单域锚点 |
| `ap2_m1` | 混合三域 | mixed | ≈0.56 F1 / ~0.61 Acc | AP2 M1 emotion_shift | — | 可选演示 | **勿与 R4 混排名** |
| `agent_chinese` | 遗留 | zh/mixed | — | AP2 中文文本微调 | — | 遗留 | 优先用 zh_agent_v2 |
| `ap4_w005` | 混合 DA | mixed | ≈0.528 F1 | AP4 DA w=0.05 | — | 对照 | 域适应消融 |
| `mosei_only` | 实验 | en | — | AP1 MOSEI | — | experimental | 单域对照 |
| `sdavt_mosei_r4` | 实验 | en | 0.6792 | R4 F_O_ES | — | experimental | MOSEI 单域；**勿中文部署** |
| `sdavt_crema_r4` | 实验 | en | Acc **0.6048** | R4 C4_C3 Warmstart | — | experimental | CREMA 单域；**勿中文部署** |

### 14.2 自动语言 → Preset 路由（产品逻辑）

| 检测语言 | `suggested_preset` | 说明 |
|----------|-------------------|------|
| 中文（zh） | `sdavt_meld_zh_agent_v2` | 与默认 `.env` 一致 |
| 英文（en） | `sdavt_meld_v3_r4` | 英文监督与 RoBERTa 对齐 |
| 用户显式选择 | 覆盖自动建议 | `checkpoint_preset` 优先 |

实现：`chinese_inference_router.py`（`suggested_preset`）+ `model_router.py`（`metadata.auto_preset`）。

### 14.3 为何「仅 MELD 做中文微调」

| 数据集 | 是否中文微调 | 理由 |
|--------|--------------|------|
| **MELD** | **是（v1→v2）** | 对话文本强依赖；换中文 BERT + 伪中文/采集注入后在线收益明确 |
| MOSEI | **否** | 英文 YouTube 字幕；换中文词表无监督对齐，ROI 低；保留 `sdavt_mosei_r4` 仅实验 |
| CREMA | **否** | 表演语料文本极弱（p4 上 T/AT collapse）；应用侧靠音频/视觉；保留 `sdavt_crema_r4` 仅实验 |

### 14.4 前端 optgroup 建议文案

```
├─ 推荐部署
│   ├─ sdavt_meld_zh_agent_v2（中文，F1=0.611）← 默认
│   └─ sdavt_meld_v3_r4（英文，F1=0.696）
├─ 中文对照
│   └─ sdavt_meld_zh_agent（v1，F1=0.601）
├─ 历史 / 混合
│   ├─ ap2_m1、meld_only、ap4_w005、agent_chinese
└─ 实验（单域）
    └─ sdavt_mosei_r4、sdavt_crema_r4、mosei_only
```

### 14.5 路径速查（相对 `PROJECT_ROOT`）

| Preset | Config | Checkpoint |
|--------|--------|------------|
| `sdavt_meld_zh_agent_v2` | `config/sdavt_v3_r4/p3_m3/meld/M3_M7_chinese_agent_v2.yaml` | `checkpoints_sdavt_v3_r4/SDAVT_R4_M3_M7_chinese_agent_v2/checkpoint_finetune_best_f1.pth` |
| `sdavt_meld_v3_r4` | `config/sdavt_v3_r4/p3_m3/meld/M3_M7_combo.yaml` | `checkpoints_sdavt_v3_r4/SDAVT_R4_M3_M7_combo/checkpoint_pretrain_best_f1.pth` |
| `sdavt_meld_zh_agent` | `config/sdavt_v3_r4/p3_m3/meld/M3_M7_chinese_agent.yaml` | `checkpoints_sdavt_v3_r4/SDAVT_R4_M3_M7_chinese_agent/checkpoint_finetune_best_f1.pth` |

训练/状态日志：`outputs_sdavt_v3_r4/status/m3m7_zh_finetune.log`、`m3m7_zh_v2_full_finetune.log`（v2：`EXIT_CODE=0`，early-stop @ep10，best@ep5）。

---

## 附录 A：原始路径速查

- 重跑日志：`project/logs_rerun/`  
- 准确率序列日志：`project/logs_accuracy_seq/`  
- **R4 日志**：`project/logs_sdavt_v3_r4/`（TB `:6008`）  
- 重跑权重：`project/checkpoints_rerun/`  
- 准确率序列权重：`project/checkpoints_accuracy_seq/`  
- **R4 权重**：`project/checkpoints_sdavt_v3_r4/`  
- R4 状态/报告：`project/outputs_sdavt_v3_r4/`  
- 根目录文献：`article_guide.md`、`dataset_application_guide.md`、`research_guide.md`  
- 项目详细实现：`project/详细文档.md`  
- Agent Preset：`emotion-agent/backend/app/core/config.py`
