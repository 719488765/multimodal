## 文档结构总览与阅读路线（2026-03 修订）

**文档定位（三合一）**：项目说明书 · 实验操作指导 · 毕业论文写作参考；同时承担 **实验过程日志**、**后续实验规划**、**可复现操作说明**。

**推荐阅读顺序**

| 目的 | 建议阅读路径 |
|------|----------------|
| **从零跑通一次** | 第一章 → 第二章 → **第八章**（分步实验）→ **第九章**（全流程 SOP：`9.1`–`9.10`） |
| **查已跑实验数字与结论** | **第十一章**「实验记录」导航表 + 小节 **`10.5`–`10.5.7`**（**以该章表格与小结为准**） |
| **理解与写「方法/模型」** | 第四章数据 → 第五章模型与融合 → 第六章训练与损失 |
| **写论文实验章与摘要** | 第十二章（成稿模板）+ 第十一章数据 + **第十三章**交付核对 |

**章节地图（全书顺序）**

| 章 | 主题 |
|----|------|
| 一 | 使用说明 |
| 二 | 项目结构与脚本 |
| 三 | 创新点与论文表述素材 |
| 四 | 数据与预处理 |
| 五 | 模型结构与融合模块 |
| 六 | 训练与损失、`train.py` |
| 七 | 日志、CSV、TensorBoard |
| 八 | 分步实验指导（原「实验一/二/三」） |
| **九** | **全流程实验步骤（环境→预训练→消融→视频扩展→论文主线清单）** |
| **十** | **从 CSV/TensorBoard 到论文图表** |
| **十一** | **实验记录与工作日志（含各次 run 结果）** |
| **十二** | **毕业论文写作与实验策略** |
| **十三** | **执行清单与交付标准** |

**约定**：第十一章内小节标题仍使用 **`10.0`–`10.6`、`10.5.x`**（与历史笔记、TensorBoard 注释一致）；**全书第几章**以本页「章节地图」为准。全流程 SOP 与「从 CSV 写论文」分别以 **第九章、第十章** 为主入口。

---

## 一、文档使用说明（给“基础薄弱”的你）

这份文档的目标是：**就算你几乎没有深度学习/工程基础，只要耐心按照本说明一步一步操作，也能完整跑通本项目的实验，并完成硕士论文的实验章节。**

为了做到这一点，本说明会覆盖：

- 项目整体结构与每个关键 `.py` 文件的作用；
- 三个开源数据集（CREMA-D、MELD、CMU-MOSEI）如何获取与整理；
- 模型结构、训练脚本、配置文件如何理解；
- 从“零开始”如何一步步启动训练、查看和保存结果；
- 如何设计和执行消融实验（尤其是模态消融）；
- 如何把整个实验过程写进硕士论文的“实验设置 / 实验结果与分析”章节。

**使用建议**：

- 第一次阅读时建议从头到尾阅读一遍，形成整体印象；
- 真正做实验时，边看边做，按照“第八章：实验操作分步指导”的顺序执行；
- 写论文时，重点参考“第十二章：论文实验部分写作框架”，逐小节填充内容。

你可以把这份文档看成是：**“项目说明书 + 实验手册 + 论文写作指南”三合一**。

---

## 二、项目整体结构与关键脚本说明

本项目代码根目录：`/home/lizhichun_24/sda1/code/multimodal/project`

下面按目录讲解你最需要关心的部分。

### 2.1 顶层结构（只列出重要部分）

- `config/`
  - `config.yaml`：**全局配置文件**，包括模型结构、训练超参数、数据集映射、模态开关等。几乎所有实验设置都在这里修改。
- `data/`
  - `dataset.py`：**数据加载核心类** `MultimodalDataset`，负责从 `data/train|val|test/...` 读取样本并拼成 batch。
  - `preprocess.py`：一些预处理工具（视频帧提取、音频处理、文本处理等）。
- `docs/`
  - `PROJECT_OVERVIEW_AND_TRAINING_PLAN.md`：本说明文档（你现在看的这个）。
  - `CMU_MOSEI_SETUP_GUIDE.md`：专门针对 MOSEI 的下载和整理指南。
  - `WEEKLY_REPORT_CMU_MOSEI.md`：你之前写的周报，可作为时间线记录。
- `models/`
  - `feature_extractors.py`：视频/音频/生理/文本四种特征提取器。
  - `multimodal_model.py`：**主模型** `MultimodalEmotionModel`，整合四种模态、注意力融合、域适应等。
  - `attention_modules.py`：标准多头注意力融合模块。
  - `emotion_shift.py`、`leader_follower_attention.py`、`two_stage_fusion.py`：不同融合策略实现。
  - `balanced_loss.py`：类不平衡损失（ClassBalancedLoss、FocalLoss 等）。
  - `domain_adaptation.py`：域对抗模块（DomainAdversarialModule）和数据集特定归一化。
- `scripts/`
  - `organize_crema_d.py`：**CREMA-D 数据集整理脚本**，从原始文件转换到统一结构。
  - `organize_meld.py`（如果有）：**MELD 整理脚本**，思路与 CREMA-D 类似。
  - `organize_cmu_mosei_from_raw.py`：**从 Kaggle 版 MOSEI `.csd` 特征构建中间目录 `CMU_MOSEI_MM` 的脚本**，提取 OpenFace2 / COVAREP / 文本 / 标签。
  - `merge_cmu_mosei_to_data.py`：**把 MOSEI 中间结果合入统一 `data/train|val|test` 的脚本**。
  - `download_cmu_mosei_sdk.py`：早期版本的 SDK 下载脚本，目前主要采用 Kaggle 版，可作为参考。
  - `train.py`：**训练入口脚本**，预训练与微调都从这里启动。
- `utils/`
  - `helpers.py`：辅助函数（配置加载、设备选择、checkpoint 保存/加载、指标计算、实验日志初始化与写入 JSON/CSV）。
- `use_data.md` / `README.md`
  - 早期项目说明和数据使用文档，可作为补充阅读。

### 2.2 各关键脚本的作用和你要做的事

下面是“文件名 → 作用 → 你需要做什么”的对照表（只列核心）：

- **`config/config.yaml`**
  - 作用：集中管理实验配置（模型结构、模态开关、训练超参、损失配置、数据集配置、路径等）。
  - 你要做的事：
    - 修改 `model.modalities` 进行模态消融（如只用文本、只用视频等）；
    - 调整训练 epoch、batch_size、learning_rate 等；
    - 控制是否开启域适应、类别平衡损失。

- **`data/dataset.py`**
  - 作用：统一从 `data/train|val|test/{video,audio,text,labels,physiological}` 读取样本，并识别样本来自哪一个数据集（CREMA/MELD/MOSEI）。
  - 你要做的事：
    - 一般不需要改代码，只要保证数据集整理脚本最终把文件放到正确目录即可。

- **`scripts/organize_crema_d.py`**
  - 作用：从 CREMA-D 原始下载文件中抽取视频/音频/文本/标签，按照统一的命名规则和目录结构保存。
  - 你要做的事：
    - 按文档中的命令运行该脚本一次，确认 CREMA-D 样本出现在 `data/{train,val,test}` 下。

- **`scripts/organize_cmu_mosei_from_raw.py`**
  - 作用：
    - 读取 Kaggle 版 MOSEI 中的 `.csd` 文件；
    - 提取视觉特征（OpenFace2）、音频特征（COVAREP）、文本信息、情感标签；
    - 保存为 `.npy` 和 `.txt` 文件到中间目录 `data/CMU_MOSEI_MM/...`。
  - 你要做的事：
    - 按文档中“数据整理步骤”执行一次，过程可能较慢，但只需一次。

- **`scripts/merge_cmu_mosei_to_data.py`**
  - 作用：
    - 将 `CMU_MOSEI_MM` 中的数据重命名为 `mosei_*` 格式；
    - 合并到最终训练目录 `data/train|val|test/...` 中，与 CREMA-D、MELD 并存。
  - 你要做的事：
    - 在上述整理完成后执行一次，确认 `data/train/video` 下有 `mosei_...npy` 文件。

- **`models/feature_extractors.py`**
  - 作用：
    - `VideoFeatureExtractor`：支持原始视频帧与 `.npy` 特征（MOSEI）。
    - `AudioFeatureExtractor`：基于 Wav2Vec2 抽取语音特征。
    - `PhysiologicalFeatureExtractor`：LSTM/CNN 处理生理信号（预留）。
    - `TextFeatureExtractor`：BERT 文本编码器。
  - 你要做的事：
    - 了解输入输出形状即可，一般不需改动。

- **`models/multimodal_model.py`**
  - 作用：
    - 组合四个特征提取器；
    - 使用多头注意力/情感转变/领导-跟随等策略融合多模态特征；
    - 输出离散情绪分类、连续情绪维度和趋势预测；
    - 实现域适应、数据集特定归一化；
    - 根据 `config.model.modalities` 中的开关决定是否使用各模态（模态消融）。
  - 你要做的事：
    - 理解大致结构即可，具体实验时重点关注模态开关的作用。

- **`scripts/train.py`**
  - 作用：
    - 统一管理训练过程（预训练/微调）；
    - 加载配置和数据集；
    - 初始化模型与优化器；
    - 调用 `train_epoch` 和 `validate` 循环；
    - 调用 `utils.helpers` 中的日志记录与 checkpoint 保存函数。
  - 你要做的事：
    - 使用命令行运行预训练和微调；
    - 查看日志输出，确认训练是否正常进行。

- **`utils/helpers.py`**
  - 作用：
    - `load_config` / `setup_device` / `save_checkpoint` / `load_checkpoint`；
    - `calculate_metrics`：计算 Accuracy、Precision、Recall、F1 等；
    - `init_experiment_logging` / `append_metrics_json` / `append_metrics_csv`：初始化和记录训练过程，生成 JSONL 与 CSV 文件。
  - 你要做的事：
    - 了解日志文件位置与格式，方便后续用 Pandas/Excel 分析。

---

## 三、研究创新点与贡献（硕士论文的两个及以上创新点）

本项目面向“开源多模态情绪数据集 + 驾驶员情绪分析”应用场景，侧重点是**工程整合 + 方法组合 + 实验平台搭建**。对于吉林大学软件学院专硕来说，这类“综合工程创新 + 定制实验框架”是完全可以作为硕士论文创新点的。下面总结出至少三个可以写入论文的创新点（你在正文里可以选其中 2–3 个作为“主要创新点”）。

### 3.1 创新点一：跨数据集的统一多模态情绪识别框架（CREMA-D + MELD + CMU-MOSEI）

**核心内容：**

- 现有大量工作只在**单一数据集**（如单独在 CREMA-D 或 MELD 或 MOSEI）上训练和评估，模型往往对某个数据集“过拟合”，难以跨场景泛化；
- 本项目将 **三个风格差异很大的开源多模态情绪数据集** 统一到同一目录结构和标签空间：
  - `project/data/{train,val,test}/{video,audio,text,labels,physiological}`；
  - 文件命名采用 `crema_*`、`meld_*`、`mosei_*` 前缀，统一情绪类别映射为 7 类；
  - 在 `MultimodalDataset` 中引入 `dataset_id` 概念，可识别样本来源数据集。
- 在模型层面引入：
  - **域对抗模块（DomainAdversarialModule）**：显式建模“来自哪个数据集（域）”，并通过梯度反转使特征“对域不敏感”；
  - **数据集特定归一化（DatasetSpecificNormalization）**：为每个数据集维护独立的归一化统计，减轻域间分布差异。
- 结合数据层面的 **BalancedDatasetSampler（平衡采样）** 与损失层面的 **类别平衡损失**，从数据、特征和损失三个角度共同缓解“跨数据集域偏移 + 类别不平衡”问题。

**可以在论文中这样表述：**

- 提出一种**面向多源开源数据集的统一多模态情绪识别框架**，通过统一的数据组织结构和标签空间，将 CREMA-D、MELD、CMU-MOSEI 三个数据集整合在同一训练流程中；
- 设计并实现了“数据层平衡采样 + 特征层域对抗 + 归一化层域特定统计 + 损失层类别平衡”的多层次跨域泛化机制，在多源数据情绪识别场景下提升了模型的鲁棒性和跨数据集泛化能力。

这属于：**“跨数据集统一建模 + 域适应 + 工程平台整合”的创新点**，非常适合软件工程/软件专硕类论文。

### 3.2 创新点二：基于高层特征的 CMU-MOSEI 自动转换与统一训练接口

**核心内容：**

- CMU-MOSEI 官方原始数据包含大量视频和音频，很多研究直接使用 SDK 读取 `.csd` 文件在自己的框架中处理；
- 本项目针对 Kaggle 版的 **高层特征 `.csd` 文件（OpenFace2、COVAREP、TimestampedWords、Labels）** 设计了一套自动转换流程：
  - `organize_cmu_mosei_from_raw.py`：从 `.csd` 文件中自动提取视觉/音频/文本/标签，并保存为 `.npy` 和 `.txt`；
  - `merge_cmu_mosei_to_data.py`：将中间结果 `CMU_MOSEI_MM` 合入统一的 `data/train|val|test` 目录中；
  - 更新 `VideoFeatureExtractor` / `AudioFeatureExtractor` / `MultimodalDataset`，使其同时支持原始视频帧和 **预提取的高层特征序列**（如 `(100,713)` 的 OpenFace2）。
- 最终实现效果：**不改主模型结构，就能直接在统一框架中复用 MOSEI 的高层特征**，大幅降低了使用 MOSEI 的门槛。

**可以在论文中这样表述：**

- 针对 CMU-MOSEI 高层特征 `.csd` 格式难以直接接入现有训练框架的问题，提出并实现了一套**自动特征转换与统一训练接口方案**，将 OpenFace2/COVAREP/文本/情绪标签转化为统一的 `.npy/.txt` 文件组织形式；
- 在视觉与音频特征提取器中设计了**兼容原始模态与高层特征的双通路结构**，使模型既可以直接处理原始视频/音频，又能复用已有高层特征，提升了系统的灵活性与可扩展性。

这属于：**“高层特征集成 + 通用接口设计”的工程与方法结合创新点**。

### 3.3 创新点三：配置驱动的模态消融与实验管理框架

**核心内容：**

- 传统模态消融实验往往需要手动修改代码（注释掉某个模态的 forward 分支），不利于系统化实验与复现实验；
- 本项目在 `config/config.yaml` 中设计了 **模态开关配置**：

  ```yaml
  model:
    modalities:
      use_video: true
      use_audio: true
      use_physiological: false
      use_text: true
  ```

- 在 `MultimodalEmotionModel` 中根据这些开关：
  - 决定是否真正调用对应模态的特征提取器；
  - 如关闭某模态，则为该模态生成全零特征张量，表示“模型完全看不到该模态信息”；
  - 不需要改任何训练/前向代码，只需改 YAML 配置即可完成模态组合切换。
- 配合 `utils.helpers` 中的 **JSONL/CSV 日志记录** 与 TensorBoard 可视化，本项目形成了一套：
  - “配置驱动的实验组合（V/A/T/VA/VT/AT/VAT）”；
  - “统一记录损失与指标曲线”；
  - “可直接用于论文图表与消融分析”的实验管理框架。

**可以在论文中这样表述：**

- 提出一种**基于配置文件的模态开关与实验管理框架**，通过简单的配置修改即可系统化地完成多模态组合实验（如 V/A/T/VA/VT/AT/VAT），避免了频繁修改代码带来的错误与复现困难；
- 结合自动化日志记录与可视化工具，为多模态情绪识别研究提供了一套可复现、可扩展的实验平台，有利于开展大规模的模态消融与参数敏感性分析。

这一点既是工程创新，也是“实验平台创新”，非常贴合软件专硕“工程+实验”的诉求。

> **论文撰写建议：**  
> 在最终论文中，你可以将上述三个创新点中的任意两个作为“主要创新点”，例如：  
> - 创新点 1：跨数据集统一多模态情绪识别框架与域适应机制；  
> - 创新点 2：基于高层特征的 CMU-MOSEI 自动转换与统一训练接口；  
> 将“配置驱动的模态消融与实验管理框架”写成“辅助创新点 / 工程创新”，放在论文创新点列表的第 3 条。

### 3.4 创新点四：多策略可插拔的多模态融合架构（统一封装多篇顶会方法）

**核心内容：**

- 现有很多工作在多模态情绪识别中只使用单一融合策略（例如简单的拼接 + MLP，或单一的 Transformer 结构），一旦想对比不同融合方法，往往需要重写模型结构，工程成本高，难以系统化评估；
- 本项目在 `MultimodalEmotionModel` 中统一封装了多种来自顶会论文的融合策略，包括：
  - 标准多头注意力融合（`MultimodalFusion`）；
  - 情感转变感知融合（`EmotionShiftFusion`，CFN-ESA 思想）；
  - 领导-跟随注意力（`MultimodalLeaderFollowerFusion`，连续情绪识别中的 leader-follower 思想）；
  - 两阶段融合（`TwoStageFusion`，GA2MIF 类似思路，先模态内再模态间）。
- 通过 `config.model.attention.fusion_strategy` 一个配置项即可切换不同融合策略，而不需要改动主体训练流程和数据接口；
- 再配合 `leader_modal`、`num_heads`、`num_gat_layers` 等参数，可以方便地在同一实验平台下对比不同融合结构在多源情绪数据上的表现。

**可以在论文中这样表述：**

- 设计了一种**多策略可插拔的多模态融合架构**，在统一的模型框架内封装了多种代表性的多模态融合方法（标准注意力、情感转变感知、领导-跟随注意力、两阶段融合等），并通过配置文件实现无缝切换；
- 该架构便于在同一数据和实现基础上系统评估不同融合策略的优劣，为后续多模态情绪识别方法研究提供了统一的实验平台。

这一创新点是在**模型架构层面**的：强调你做的是“一个可以装下多种融合算法的统一骨架”，而不是单一模型。

### 3.5 创新点五：离散与连续情绪联合建模及可选的功能相关性约束

**核心内容：**

- 传统情绪识别工作往往只关注**离散情绪分类**（happy/sad/angry/...），或者只做**连续情绪回归**（valence/arousal），很少在一个统一模型中同时联合建模两者，更少考虑趋势预测与模态间功能相关性约束；
- 本项目在 `MultimodalEmotionModel` 中设计了**共享特征、多个头（heads）的联合建模结构**：
  - 共享的融合表示 `fused_features`；
  - 基于同一特征的离散情绪分类头（`emotion_classifier`）；
  - 基于同一特征的连续情绪维度回归头（`emotion_regressor`，valence/arousal）；
  - 可选的情绪趋势预测头（`trend_predictor`），用于建模情绪随时间的变化趋势（特别适合视频/对话场景）。
- 此外，模型中预留了可选的**功能最大相关损失（MultimodalCorrelationLoss，简称 FMC Loss）**：
  - 在预训练阶段通过最大奖励不同模态特征间的功能相关性，鼓励模型学习到“在情绪任务上协同变化”的模态表示；
  - 这有助于提升多模态之间的一致性和互补性，尤其在某些模态缺失或噪声较大的情况下。

**可以在论文中这样表述：**

- 提出一种**离散情绪分类与连续情绪维度联合建模的多头结构**，在共享多模态融合表示的基础上，同时优化分类、回归和趋势预测任务，从而捕捉更丰富的情绪信息；
- 在此基础上引入可选的**多模态功能相关性约束**，通过辅助损失增强不同模态之间的表征一致性，提高模型在多源异构情绪数据集上的鲁棒性和泛化能力。

这个创新点偏向**算法/任务建模层面**：强调你不只是“做分类”，而是在一个统一模型里，联合处理了离散 + 连续 + 趋势三个情绪任务，并考虑了模态间的相关性约束。

---

### 3.6 可直接用于论文的“创新点”章节示例

下面是一段已经整理好的论文“创新点”章节示例文字，你可以在最后定稿时根据学校格式稍作调整后，**直接复制到学位论文中使用**。

> **本论文的主要创新点概括如下：**  
>  
> （1）**构建了面向多源开源数据集的统一多模态情绪识别框架。** 针对现有情绪识别研究多在单一数据集上建模、跨场景泛化能力有限的问题，本文统一整合了 CREMA‑D、MELD、CMU‑MOSEI 三个风格差异明显的多模态情绪数据集，设计了统一的数据组织结构与标签空间，并在此基础上引入平衡采样、域对抗模块（DomainAdversarialModule）以及数据集特定归一化（DatasetSpecificNormalization）等机制，从数据层、特征层和损失层多维度缓解“域偏移 + 类别不平衡”问题，提升了模型在多源情绪数据上的鲁棒性和跨数据集泛化能力。  
>  
> （2）**提出了基于高层特征的 CMU‑MOSEI 自动转换与统一训练接口。** 针对 CMU‑MOSEI 数据集中高层特征 `.csd` 文件难以直接接入通用训练框架的问题，本文设计并实现了 `organize_cmu_mosei_from_raw.py` 与 `merge_cmu_mosei_to_data.py` 等脚本，将 OpenFace2 视觉特征、COVAREP 声学特征、时间戳文本和情绪标签自动转换为统一的 `.npy/.txt` 形式，并与 CREMA‑D、MELD 一同组织到 `project/data/train|val|test/{video,audio,text,labels,physiological}` 目录结构下。同时，在视觉与音频特征提取器中设计了兼容原始模态与高层特征的双通路结构，使模型既可以直接处理原始视频/音频，又能复用已有高层特征，大幅降低了使用 CMU‑MOSEI 的工程门槛。  
>  
> （3）**构建了多策略可插拔的多模态融合架构与联合情绪建模机制。** 本文在统一的 `MultimodalEmotionModel` 框架中封装了多种代表性的多模态融合方法，包括标准多头注意力融合（MultimodalFusion）、情感转变感知融合（EmotionShiftFusion，借鉴 CFN‑ESA 思想）、领导‑跟随注意力融合（MultimodalLeaderFollowerFusion，借鉴 Leader‑Follower 思想）以及两阶段图注意力 + 跨模态注意力融合（TwoStageFusion，借鉴 GA2MIF 思想），并通过配置项 `fusion_strategy` 实现无缝切换。在此基础上，本文采用共享融合表示、多头输出的结构，同时进行离散情绪分类、连续情绪维度回归与情绪趋势预测，并引入多模态功能最大相关损失（MultimodalCorrelationLoss，借鉴 MFMC 思想）作为可选辅助约束，从而在一个统一模型中联合建模多种情绪表征形式，增强了模态间表征一致性与互补性。  
>  
> （4）**实现了配置驱动的模态消融与实验管理平台。** 针对多模态情绪识别中模态组合实验难以系统组织和重复利用的问题，本文在配置文件中设计了 `use_video/use_audio/use_text/use_physiological` 等模态开关，并在模型内部实现了与之对应的特征屏蔽机制，使得研究者无需修改代码，仅通过修改配置即可方便地完成 V/A/T/VA/VT/AT/VAT 等多种模态组合实验。结合自动化 JSONL/CSV 日志记录与 TensorBoard 可视化工具，本文搭建了一套可复现、可扩展的多模态实验管理平台，为后续开展大规模模态消融和参数敏感性分析提供了良好的工程基础。

---

## 四、数据与预处理流程（从“下载到可训练”的完整路径）

### 4.1 数据集来源与统一结构

1. **CREMA-D**
   - 实验室环境，演员表演情绪，视频格式多为 `.flv`。
   - 通过脚本 `scripts/organize_crema_d.py` 解析原始文件和标签，生成统一结构：
     - `train/val/test/video/crema_*.flv`
     - `train/val/test/audio/crema_*.wav`（如有）
     - `train/val/test/text/crema_*.txt`（可能为占位文本）
     - `train/val/test/labels/crema_*.txt`（离散情感标签 + 可能的维度标签）。

2. **MELD**
   - Friends 剧集中语音对话片段，多说话人、多轮对话，视频 `.mp4` + 对话文本。
   - 通过对应整理脚本（类似 CREMA-D）将原始数据转为统一结构：
     - `train/val/test/video/meld_*.mp4`
     - `train/val/test/audio/meld_*.wav`
     - `train/val/test/text/meld_*.txt`
     - `train/val/test/labels/meld_*.txt`。

3. **CMU-MOSEI**
   - YouTube 视频中的自然语言评论片段，多样化真实场景。
   - 你采用的是 **Kaggle 打包的 MOSEI 特征版本**：包含多种 `.csd` 高层特征文件（视觉 OpenFace2、声学 COVAREP、文本、情感标签等），而非原始 `.mp4/.wav`。
   - 通过 `scripts/organize_cmu_mosei_from_raw.py` 完成以下处理：
     - 从 `CMU_MOSEI_VisualOpenFace2.csd` 提取视觉序列特征，统一采样为长度约 100 帧，保存为：
       - `data/CMU_MOSEI_MM/{train,val,test}/video/mosei_*_*.npy`，形状约为 `(100, 713)`。
     - 从 `CMU_MOSEI_COVAREP.csd` 提取声学特征，同样统一采样为 `(100, feature_dim)` 的 `.npy` 文件。
     - 从 `CMU_MOSEI_TimestampedWords.csd` 提取或构造文本（目前实现为简化/占位版本，可后续精细化）。
     - 从 `CMU_MOSEI_Labels.csd` 解析情感分数，映射为离散情感（happy / sad / angry / fear / neutral / surprise / disgust）+ 连续维度（valence/arousal）并写入 label 文本。
   - 再通过 `scripts/merge_cmu_mosei_to_data.py` 把 `CMU_MOSEI_MM` 合并进统一的 `data/train|val|test` 目录中，与 CREMA-D、MELD 放在一起。

### 4.2 核心数据加载类 `MultimodalDataset`

文件：`data/dataset.py`。

关键功能：

- 扫描 `data/{split}/{video,audio,text,labels,physiological}` 目录。
- 通过文件名前缀识别数据集：`crema_* → dataset_id=0`，`meld_* → 1`，`mosei_* → 2`。
- 支持视频、音频、生理信号、文本四种模态；同时支持 **原始文件** 和 **预提取 `.npy` 特征**：  
  - `SUPPORTED_VIDEO_FORMATS = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.npy')`  
  - `SUPPORTED_AUDIO_FORMATS = ('.wav', '.mp3', '.flac', '.m4a', '.npy')`
- `_load_video` / `_load_audio` 中：
  - 如果后缀是 `.npy`，使用 `np.load → torch.from_numpy().float()`，直接读入特征序列；
  - 否则使用 `cv2.VideoCapture` / `librosa.load` 从原始 `.mp4/.wav` 中采样帧 / 波形。
- 文本通过 `_load_text` 读取 `.txt`，再在 `DataPreprocessor` 中完成 BERT tokenizer 处理。
- 标签解析：
  - 情感类别（字符串）映射到统一标签空间，见 `config/config.yaml` 的 `datasets.*.emotion_map`；
  - 连续维度（valence/arousal）解析为二维回归目标。

因此：**不同数据集的异构数据 → 统一转为 `torch.Tensor` 格式的 video/audio/text/label**，并统一返回一个 `batch` 字典，供模型前向使用。

---

## 五、模型结构与特征提取（从“输入数据”到“情绪预测”）

### 5.1 特征提取器 `models/feature_extractors.py`

四个主要模块：

1. `VideoFeatureExtractor`
   - 支持四种输入：
     - `(B, T, C, H, W)`：时序视频帧 → ResNet-50 backbone → `cnn_projection` → `(B, T, 512)`；
     - `(B, C, H, W)`：单帧图像 → ResNet-50 → `cnn_projection` → `(B, 512)`；
     - `(B, T, F)`：**预提取视觉特征序列**（如 MOSEI OpenFace2 `(100, 713)`）→ `feature_projection(Linear(F,512))` → `(B, T, 512)`；
     - `(B, F)`：单向量特征 → `feature_projection` → `(B, 512)`。
   - 这保证了：
     - CREMA-D / MELD 使用 ResNet-50 从原始视频帧提特征；
     - MOSEI 使用预先抽取的 OpenFace2 特征，直接线性映射到 512 维，不再重复 CNN，既利用了官方特征，又避免重复重算，整体效率更高。

2. `AudioFeatureExtractor`
   - 使用 HuggingFace `Wav2Vec2Model` 提取语音语义特征：
     - 输入 `audio_waveform (B, T)` 为波形采样点；
     - 输出 `hidden_states (B, T, hidden_dim)` → `projection (Linear)` → `(B, T, 512)` → 时序平均池化 → `(B, 512)`。
   - 当前版本对 MOSEI 的 `.npy` 声学特征暂仍按“波形”形式送入 Wav2Vec2（数值上可行），后续如做更细致的声学特征建模，可以增设专门的 COVAREP extractor 或轻量 MLP 头，用于效率对比实验。

3. `PhysiologicalFeatureExtractor`
   - 针对 EEG/ECG/GSR 等时序生理信号，采用 Bi-LSTM 或 1D-CNN 提取特征，输出 `(B, 512)`。
   - 在当前三数据集中（CREMA-D/MELD/MOSEI）尚未用到，但为以后接入真实驾驶生理信号做好接口。

4. `TextFeatureExtractor`（在 `models/multimodal_model.py` 中集成）
   - 使用 BERT (`bert-base-uncased`) 做文本编码，输出 `(B, 512)` 的句向量。

### 5.2 多模态融合与输出（`models/multimodal_model.py`）

整体结构（简要）：

- 各模态 extractor 得到：
  - `v ∈ R^{B×512}`, `a ∈ R^{B×512}`, `p ∈ R^{B×512}`, `t ∈ R^{B×512}`。
- 融合模块（`model.attention` 配置）：
  - 支持多头注意力、多层 Transformer、情感转变感知（emotion_shift）、领导-跟随注意力（leader_follower）、两阶段融合（GAT 等）。
  - 当前实验主线推荐策略：`fusion_strategy: "standard"`（保证与既有 AT/T/A/V/VT/AVT 消融可比）；`emotion_shift`、`leader_follower`、`two_stage` 作为后续结构消融逐步开启。
- 输出层：
  - 情感分类：`emotion_logits`（维度为 `model.output.emotion_classes`，默认为 7 类）；
  - 情感维度回归：`emotion_dimensions`（valence, arousal）；
  - 趋势预测：`trend_prediction`（可选，用于预测情绪随时间变化趋势）。

### 5.3 顶会融合模块及本项目改进详解

本项目在融合阶段参考了多篇顶会方法，并在统一框架下作了**多模态扩展与工程简化**。下面分别介绍这些模块在原论文中的思想、在本项目中的具体实现，以及我们的改进点。这一部分内容可以直接用于你论文中“方法”章节里对各子模块的介绍和分析。

#### 5.3.1 标准多头注意力融合 `MultimodalFusion`（基础架构）

- **对应文件**：`models/attention_modules.py` → `MultimodalFusion`。
- **原理与结构**：
  - 先对每个模态得到固定长度特征：
    - 视频如果是序列 `(B,T,512)`，先用 `TemporalAttention` 按时间加权，得到 `(B,512)`；
    - 音频/生理/文本本身就是 `(B,512)`。
  - 将四个模态特征堆叠成 `(B, num_modalities=4, hidden_dim=512)`；
  - 经过 `L` 层“Transformer Encoder 样式”的模块：
    - 每层包含一层自注意力 `MultiHeadSelfAttention`：
      - 在“模态维度”上做注意力（4 个模态互相看对方），建模模态间依赖；
      - 残差连接 + LayerNorm；
    - 一层前馈网络（FFN）：
      - `Linear(512→2048) + GELU + Dropout + Linear(2048→512) + Dropout`；
      - 残差连接 + LayerNorm。
  - 最后对模态维度做平均池化，得到融合特征 `fused_features ∈ R^{B×512}`。
- **本项目的作用和改进要点**：
  - 作为所有复杂融合策略（EmotionShift、LeaderFollower、TwoStage）的**基线骨架**，便于在同一前提下比较效果与效率；
  - 通过 `config.model.attention.num_layers/num_heads` 可以轻松调节层数和头数，配合后续效率实验（例如“1 层 vs 3 层”、“4 头 vs 8 头”），分析性能-效率折中；
  - 将时序信息通过 `TemporalAttention` 先压缩，再做模态注意力，既保留部分时间信息，又避免时间×模态的双重注意力带来的巨大计算量，属于一种**轻量化的时序+模态联合建模**策略。

#### 5.3.2 情感转变感知融合 `EmotionShiftFusion`（CFN‑ESA 思想的多模态扩展）

- **对应文件**：`models/emotion_shift.py`。
- **原始 CFN‑ESA 思想（对话情绪识别）**：
  - 对话中情绪是随时间演变的，存在“情绪转折点”；
  - 用一个情绪状态编码器预测每个时间步的情绪分布，再用 LSTM 捕捉情绪序列的动态变化，将“情绪转变信息”反馈给当前时刻的特征。
- **本项目中的具体结构**：
  - `EmotionShiftAwareness`：
    - 输入：时序特征 `features (B, T, hidden_dim)`（这里是加权融合后的多模态特征）；
    - 用 `emotion_encoder` 预测每个时间步的情绪 logits → softmax 得到情绪概率；
    - 将情绪概率序列送入双向 LSTM `shift_detector`，抽取“情绪变化模式”；
    - 将原始特征与 LSTM 输出的转变特征拼接，经 `shift_fusion` 映射回 `hidden_dim`；
    - 通过相邻时间步情绪概率的差异计算 `shift_weights`，衡量情绪变化强度。
  - `EmotionShiftFusion`：
    - 先对四模态特征做加权和：`weighted_feat = Σ modal_weights[i] * feat_i`，其中 `modal_weights` 为可学习参数，文本权重可偏高；
    - 将 `weighted_feat` 输入 `EmotionShiftAwareness`，得到情绪增强特征和情绪 logits/shift_weights；
    - 再用一层 `nn.MultiheadAttention`，以文本模态为 query，以其他模态（视频、音频、生理）的堆叠表示为 key/value，做一次跨模态细化；
    - 残差 + LayerNorm 后输出最终时序融合特征，若是单帧则压缩掉时间维度。
- **相对原始 CFN‑ESA 的改进与创新**：
  - 从“单一对话模态”扩展为 **四模态联合情绪转变感知**：视频/音频/生理/文本共同参与情绪变化的建模，更贴合智能驾驶与自然视频场景；
  - 引入可学习模态权重 `modal_weights`，使模型能自动调整“文本主导”或“视听主导”等策略；
  - 将情感转变模块与跨模态注意力融合在一个统一模块中，并通过 `fusion_strategy="emotion_shift"` 配置可插拔，方便与其他策略对比；
  - 逻辑上形成“模态加权 → 情绪转变增强 → 跨模态注意力”的三级结构，既考虑时间动态，又兼顾多模态互补。

#### 5.3.3 领导‑跟随注意力融合 `MultimodalLeaderFollowerFusion`（Leader‑Follower 思想扩展）

- **对应文件**：`models/leader_follower_attention.py`。
- **原始 Leader‑Follower 思想（连续情绪识别）**：
  - 在多模态中选定一个更可靠的模态作为“Leader”（领导者），另一模态作为“Follower”（跟随者）；
  - 用 Leader 的信息通过注意力机制引导 Follower 特征更新，从而突出更可靠模态的作用。
- **本项目中的具体结构**：
  - `LeaderFollowerAttention`：
    - Leader 经 `leader_q` 映射为 Query，Follower 经 `follower_k/follower_v` 映射为 Key/Value；
    - 注意力得分 `scores = K Q^T / sqrt(d)` 表示 Follower 在不同 Leader 时刻的注意程度；
    - 用注意力加权 Leader 信息并加到 Follower 上（残差 + `out_proj`），得到增强后的 Follower 特征。
  - `MultimodalLeaderFollowerFusion`：
    - 默认把文本 `text_feat` 当 Leader，视频/音频/生理分别作为三个 Follower；
    - 对每个 Follower 分别应用 `LeaderFollowerAttention(text → follower)`；
    - 将 `[text_feat, enhanced_video, enhanced_audio, enhanced_physiological]` 在特征维拼接，经 `final_fusion + LayerNorm` 得到融合特征；
    - 同时支持单帧和时序输入。
- **相对原始 Leader‑Follower 的改进与创新**：
  - 将原本主要针对双模态（视听）的思想推广到 **四模态统一架构**，可灵活设置 Leader（未来可通过 `leader_modal` 切换为生理/视觉等）；
  - 把领导‑跟随机制实现为与其他融合策略同接口的模块，可在同一数据与训练脚本下直接对比不同融合方式，不需重写模型；
  - 在工程上兼顾“时序 + 单帧”的兼容，实现了对对话、短片段和驾驶过程不同类型输入的统一建模。

#### 5.3.4 两阶段融合 `TwoStageFusion`（GA2MIF 思想的简化与工程化实现）

- **对应文件**：`models/two_stage_fusion.py`。
- **原始 GA2MIF 思想**：
  - 第 1 阶段：通过图注意力网络（Graph Attention Network, GAT）在模态/说话人/上下文节点之间传播信息，建模结构关系；
  - 第 2 阶段：通过多头注意力机制进行精细的跨模态融合。
- **本项目中的两阶段结构**：
  - 阶段一：简化版 `GraphAttentionLayer`
    - 使用 PyTorch 内置 `nn.MultiheadAttention` 实现“节点之间的自注意力”，再配合 LayerNorm 和 Dropout，模拟 GAT 的核心思想；
    - 将四模态视为图的四个“节点”，对每个时间步分别应用图注意力，实现“模态图上的信息传播”；
    - 多层 `gat_layers` 堆叠，使每个模态的特征中逐渐编码其他模态的上下文信息。
  - 阶段二：自注意力 + 跨模态注意力
    - 对每个模态分别用 `MultiHeadSelfAttention` 做模态内部时序建模（捕捉模态内时间依赖）；  
    - 使用多层 `CrossModalAttention` 以文本为 query，以视频/音频/生理为 key/value，完成多轮跨模态交互；
    - 残差 + LayerNorm + 前馈网络，使四个模态在第二阶段进一步耦合；
    - 最后拼接四模态，再用 `final_fusion` 压缩到 `hidden_dim=512`，形成最终融合表示。
- **相对原始 GA2MIF 的改进与创新**：
  - 使用 `MultiheadAttention` 替代完整 GAT 实现，大幅减少了特殊图结构代码与计算开销，**更易于工程实现和调试**；
  - 针对“4 模态固定节点”的场景进行简化，不再需要复杂的动态图构建，从而适合在你的统一数据结构中直接使用；
  - 与前述其他融合策略（标准注意力、EmotionShift、LeaderFollower）共享统一接口和训练流程，使 GA2MIF 风格的两阶段融合真正变成“可插拔模块”，便于与其他方法在同等条件下做量化对比；
  - 将两阶段结构应用于“高层特征 + 原始模态混合输入”的场景，为多源异构特征的融合提供了一个轻量的参考实现。

#### 5.3.5 多模态功能最大相关 `MultimodalCorrelationLoss`（MFMC 思想的辅助损失）

- **对应文件**：`models/functional_correlation.py`。
- **原始 MFMC 思想**：
  - 在某个投影空间中最大化不同模态特征之间的相关性（通常通过协方差矩阵的范数/特征值）；
  - 通过这种“功能最大相关”鼓励各模态在任务上协同变化，提升情绪识别性能。
- **本项目中的实现细节**：
  - 为每个模态（video/audio/physiological/text）设计一个小投影网络（Linear → ReLU → Dropout），将 `hidden_dim=512` 映射到 `num_projections` 维；
  - 对任意两模态投影特征 `(B, num_projections)`，先中心化，再计算协方差矩阵 `cov`，最后取 Frobenius 范数作为相关性度量；
  - 累加视频‑音频、视频‑文本、音频‑文本、生理‑视频/音频/文本等多对模态之间的相关性，得到 `total_correlation`；
  - 损失定义为 `correlation_loss = -total_correlation * weight`，训练时与分类/回归损失一同最小化，相当于最大化模态之间的功能相关性。
- **在本项目中的作用和优势**：
  - 作为一个**可选的辅助损失**，通常在预训练阶段打开，在微调阶段关闭，既发挥“对齐模态表征”的作用，又控制计算代价和过拟合风险；
  - 相关性字典会记录每一对模态的相关性值，可以写入日志，用于论文中“训练过程中模态相关性的演化”分析；
  - 特别适合你当前这种“多源数据 + 预提取特征 + 原始模态混合”的复杂情形，有助于缓解某些模态噪声大、缺失或不稳定时的性能波动。

> 综上，本项目不是简单“照搬”某一篇顶会方法，而是将多种代表性多模态融合/相关性建模方法在一个统一框架内做了**工程级整合、模态扩展与轻量化改造**，这正是第三章中“创新点四/五”在模型和算法层面的具体技术支撑。

---

## 六、训练与损失设计（模型如何“学会”情绪识别）

### 6.1 综合损失 `MultimodalLoss`（`scripts/train.py`）

`MultimodalLoss` 负责将多个任务和正则项组合：

- 分类损失：
  - 标准 `CrossEntropyLoss`；或
  - `ClassBalancedLoss`（处理类别不平衡）；或
  - `FocalLoss`（着重难分类样本）。
- 回归损失：`MSELoss`，用于 valence/arousal。
- 趋势损失：`MSELoss`，用于趋势预测。
- 域适应损失：
  - 针对 `DomainAdversarialModule` 的域分类 logits，使用 `CrossEntropyLoss`，权重为 `domain_loss_weight`。

总体损失为：

$$
L = w_{cls} L_{cls} + w_{reg} L_{reg} + w_{trend} L_{trend} + \lambda_{da} L_{domain}
$$

各权重在 `config/config.yaml` 中配置：

- `training.loss_weights.classification`、`regression`、`trend`；
- `training.loss.domain_loss_weight`；
- `training.loss.use_class_balanced` / `use_focal_loss` 用于切换损失形式。

### 6.2 域适应与数据集归一化

在 `config.yaml` 中：

- `model.domain_adaptation.enabled: true`：开启域适应；
- `model.domain_adaptation.num_domains: 3`：三个域（CREMA-D, MELD, MOSEI）；
- `training.loss.use_domain_adaptation: true`：训练时加入域分类损失；
- `model.dataset_normalization.enabled: true`：为不同数据集维护分布统计，降低域偏移。

训练时：

- `MultimodalDataset` 在 `batch` 中返回 `dataset_id`；
- 模型前向时可输出 `domain_logits`，被 `MultimodalLoss` 使用；
- 梯度反转层（GRL）实现对抗训练，使共享表征对数据集 ID 不敏感，从而提升跨域泛化。

### 6.3 训练脚本与策略 `scripts/train.py`

`train.py` 的主流程：

1. 解析命令行参数：
   - `--config`: 配置文件路径（例如 `config/config.yaml`）；
   - `--mode`: `pretrain` 或 `finetune`；
   - `--resume`: 从检查点恢复训练；
   - `--dataset`: 可选，强制使用某一个数据集的标签数。
2. 加载配置与设备：`load_config` + `setup_device`。
3. 初始化数据集与 DataLoader：
   - `MultimodalDataset(data_dir, split='train'/'val')`；
   - 若 `training.sampling.enabled = true`，使用 `BalancedDatasetSampler` 做数据集平衡采样。
4. 初始化模型 `MultimodalEmotionModel` 和优化器 `AdamW` + `CosineAnnealingLR`。
5. 根据 `mode` 和配置决定是否冻结 backbone（常用于微调阶段）。
6. 进入 epoch 循环：
   - `train_epoch`: 前向 → 计算多任务损失 → 反向 → 梯度裁剪 → 更新参数；
   - `validate`: 计算验证损失和分类指标（accuracy/precision/recall/F1）；
   - 学习率调度、保存 best/定期 checkpoint。

---

## 七、训练过程数据跟踪与可视化方案（为论文准备数据支撑）

为了满足硕士论文实验的记录需求，本项目已经增加了 **详细的训练过程记录与可视化方案**。

### 7.1 本地日志文件（JSONL + CSV）

新增工具函数（`utils/helpers.py`）：

- `init_experiment_logging(config)`：
  - 根据 `paths.log_dir`（默认 `logs/`）与 `experiment.name`（例如 `"multimodal_emotion_recognition"`）创建带时间戳的实验目录：  
    `logs/multimodal_emotion_recognition_YYYYMMDD_HHMMSS/`
  - 返回：
    - `log_dir`
    - `metrics_json_path` → `metrics.jsonl`
    - `metrics_csv_path` → `metrics.csv`
- `append_metrics_json(path, record)`：将字典追加为一行 JSON（JSONL 格式），便于后期用 Python 逐行读取、统计。
- `append_metrics_csv(path, record)`：将度量记录写入 CSV，表头包括：
  - `epoch, phase, loss, cls_loss, reg_loss, domain_loss, trend_loss, accuracy, precision, recall, f1`。

在 `train.py` 中，主循环每个 epoch 后会依次写入：

- **训练阶段记录（phase="train"）**：
  - `loss`：总训练损失；
  - `cls_loss`、`reg_loss`、`domain_loss`、`trend_loss`：各构成项平均值；
  - 后续可根据需要扩展更多字段。
- **验证阶段记录（phase="val"）**：
  - `loss`：验证总损失；
  - `accuracy`、`precision`、`recall`、`f1`：分类指标。

你可以在任何时候使用 Python / Excel / Pandas 对这些 CSV/JSONL 文件做统计和绘图，用于论文中的学习曲线、损失对比、收敛速度分析等。

### 7.2 TensorBoard 可视化（本地浏览器查看）

在 `train.py` 中已集成 `SummaryWriter`：

- 依据 `config.experiment.use_tensorboard`（当前为 `true`）决定是否启用；
- 日志目录与 JSON/CSV 相同（`log_dir`）；
- 每个 epoch 写入：
  - `train/loss_total`、`train/loss_classification`、`train/loss_regression`、`train/loss_domain`、`train/loss_trend`；
  - `val/loss_total`、`val/accuracy`、`val/precision`、`val/recall`、`val/f1`；
  - 可在后续实验中按需要添加新的标量。

**使用步骤**：

1. 启动训练，例如：

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project
conda activate myenv310
python scripts/train.py --config config/config.yaml --mode pretrain
```

2. 训练开始后，记下终端输出的 `Experiment log directory: logs/...`。

3. 在服务器上（或本地）启动 TensorBoard：

```bash
cd /home/lizhichun_24/sda1/code/multimodal/project
tensorboard --logdir logs --port 6006
```

4. 在本地浏览器中访问（假设你通过 ssh 端口转发或本地运行）：

```text
http://localhost:6006
```

5. 即可看到：
   - 训练/验证损失曲线；
   - 各项评估指标随 epoch 的变化；
   - 不同实验 run 的对比（如果你多次训练）。

如果你在远程服务器上训练，可以使用 ssh 端口转发：

```bash
ssh -L 6006:localhost:6006 your_user@your_server_ip
```

然后在本地浏览器访问 `http://localhost:6006` 即可。

---

## 八、从“跑实验”角度的详细指导（一步一步完成全部实验）

本章从“操作视角”出发，你按顺序做，就能跑完预训练、微调、模态消融，并得到论文可用的数据。

### 8.1 实验一：三数据集混合预训练（先无域适应 Baseline，再加域适应做消融）

**总目标**：学习跨场景、跨数据集的通用情感表征，提高泛化能力。  
**实际操作建议**：为了先保证训练流程稳定、结果可复现，推荐分两阶段完成本实验：

- **阶段 A：无域适应的基础 Baseline（当前你正在跑的版本）**
- **阶段 B：在相同设置上打开域适应，对比“有 / 无 域适应”的提升（作为消融实验）**

#### 8.1.1 阶段 A：无域适应 Baseline（先把训练完整跑通）

1. 配置（当前实际运行版本，建议保留为“官方基线”）：
   - 预训练与数据集：
     - `training.pretrain.enabled: true`
     - `training.pretrain.datasets: ["crema", "meld", "mosei"]`
   - 模态开关（由于视频分支显存与梯度问题，当前先以“音频+文本 AT 组合”作为稳定基线，后续再逐步加回视频模态做消融）：
     - `model.modalities.use_video: false`
     - `model.modalities.use_audio: true`
     - `model.modalities.use_text: true`
     - `model.modalities.use_physiological: false`
   - 融合策略与损失：
     - `model.attention.fusion_strategy: "standard"`（标准多头注意力作为 Baseline，EmotionShift 等高级策略留到后续消融）
     - `training.loss.use_class_balanced: true`
     - `training.loss.use_focal_loss: false`
   - 域适应相关（当前关闭，等待基础训练完全稳定后再开启对比）：
     - `model.domain_adaptation.enabled: false`
     - `training.loss.use_domain_adaptation: false`
   - 为解决显存问题而采用的工程折中（建议在论文中简单说明）：
     - `training.batch_size: 4`
     - `data.video.frame_size: 160`
     - `data.video.num_frames: 8`

2. 训练命令（不变）：

```bash
python scripts/train.py --config config/config.yaml --mode pretrain
```

3. 记录与观察要点：
   - 使用 TensorBoard 和 `logs/.../metrics.csv` 记录训练过程；
   - 重点关注：
     - `train/loss_total`、`val/loss_total` 的收敛情况；
     - `val/accuracy`、`val/f1` 的最终水平；
   - 将“**AT 组合（音频+文本）、无域适应、标准注意力、类别平衡损失**”的结果视为**当前跨数据集统一建模的基础 Baseline**，后面所有改进（重新加入视频模态、开启域适应、切换融合策略、调整损失等）都与它对比。

#### 8.1.2 阶段 B：加入域适应的对比实验（消融）

在阶段 A 跑通、验证曲线稳定后，再进行**“开启域适应”的版本**：

1. 配置改动（在同一份 `config/config.yaml` 基础上）：
   - `model.domain_adaptation.enabled: true`
   - `training.loss.use_domain_adaptation: true`
   - 其余保持与阶段 A 一致（包括 `training.pretrain.datasets`、`batch_size`、`loss_weights` 等），保证公平对比。

2. 训练命令（仍然相同）：

```bash
python scripts/train.py --config config/config.yaml --mode pretrain
```

3. 记录与对比要点：
   - 使用不同的 `logs/.../` 目录区分“无域适应 / 有域适应”两个 run；
   - 对比：
     - 验证集 `F1`、`Accuracy` 是否有提升；
     - 收敛速度和训练稳定性（loss 曲线是否更平滑）；
   - 在论文中，这一部分可以写成“域适应模块的消融实验”，对应本文档 **第八章 8.1.2**（阶段 B）与 **第九章 9.4**（AT+DA vs noDA 流程重述）。

### 8.2 实验二：针对某一数据集的微调（例如 CMU-MOSEI）

**目的**：在预训练基础上进一步优化单数据集性能，验证迁移效果。

方案 A（当前配置，三数据集合并微调）：

```bash
python scripts/train.py --config config/config.yaml --mode finetune --resume checkpoints/checkpoint_pretrain_best.pth
```

方案 B（只针对 MOSEI 微调，需要在 `config.yaml` 中把 `training.finetune.datasets` 改为 `["mosei"]`，并在 `MultimodalDataset` 中加入按数据集过滤逻辑）：

```bash
python scripts/train.py --config config/config.yaml --mode finetune --dataset mosei --resume checkpoints/checkpoint_pretrain_best.pth
```

可在论文中对比：

- 只用 MOSEI 训练 vs 预训练 + MOSEI 微调；
- 有 / 无 域适应（对应 7.1 中阶段 A/B 的设置，在同一微调脚本中复用）；
- 有 / 无 类别平衡损失。

### 8.3 实验三：模态消融实验（模态开关的实际用法）

**目的**：评估不同模态（视频、音频、文本、生理）的贡献。

实现思路：

- 已经在 `config/config.yaml` 中增加模态开关：

  ```yaml
  model:
    modalities:
      use_video: true
      use_audio: true
      use_physiological: false
      use_text: true
  ```

- 在 `MultimodalEmotionModel` 内部根据这些开关决定：
  - 是否真正调用对应模态的特征提取器；
  - 如关闭某模态，则为该模态生成全零特征张量，表示“模型完全看不到该模态信息”。

- 这样你只需要改配置文件，就可以做不同模态组合的实验，而不用改代码。
- 建议设计实验组合：
  - `V`（仅视频）、`A`（仅音频）、`T`（仅文本）
  - `VA`、`VT`、`AT`、`VAT`。

每种配置训练一个 run，通过 TensorBoard 和 CSV 比较性能差异，撰写在论文“模态贡献分析”中。

**具体操作步骤示例：**

1. 仅文本模态（`T`）
   - 修改 `config/config.yaml` 中的：

     ```yaml
     model:
       modalities:
         use_video: false
         use_audio: false
         use_physiological: false
         use_text: true
     ```

   - 运行预训练或微调命令（示例）：

     ```bash
     python scripts/train.py --config config/config.yaml --mode pretrain
     ```

   - 对应的日志目录会包含仅文本模态下的损失与指标曲线，可与其他配置对比。

2. 视频 + 文本（`VT`）

   ```yaml
   model:
     modalities:
       use_video: true
       use_audio: false
       use_physiological: false
       use_text: true
   ```

3. 视频 + 音频 + 文本（`VAT/AVT`，可选组合）

   ```yaml
   model:
     modalities:
       use_video: true
       use_audio: true
       use_physiological: false
       use_text: true
   ```

4. 完整模态（包含未来生理信号 `VAPT`）
   - 在接入真实驾驶生理数据后，将 `use_physiological: true` 并保证 `MultimodalDataset` 返回对应张量，即可扩展到四模态实验。

建议在论文中至少报告以下组合的结果，并给出表格和分析：

- 单模态：`T`、`V`、`A`；
- 双模态：`VT`、`AT`、`VA`；
- 三模态：`VAT/AVT`（建议作为扩展实验，不再表述为“当前默认”）。

---

## 九、全流程实验步骤指导（从环境到视频模态拓展）

本节将之前各小节中分散的实验说明整合成一份“**端到端实验操作清单**”，从环境准备、数据整理、基线预训练、模态消融、域适应消融，到单数据集微调与后续加入视频模态的扩展实验，形成一条你后续都可以直接重复使用的完整流程。  
为避免遗忘，本节会明确标注“**已完成**”与“**待完成**”步骤。

> 建议用法：真正做实验时，优先参考本节作为“总路线图”，再在需要时回看第 5 / 10 章中的详细说明。

### 9.1 环境与数据准备（已完成一次，可复用）

1. **环境创建与基础依赖安装**（对应环境搭建章节）  
   - 按 `EXPERIMENT_ENV_SETUP.md` 创建 Conda 环境，安装 PyTorch、TorchVision、Transformers、TensorBoard 等依赖。  
   - 确认项目根目录为：`/home/lizhichun_24/sda1/code/multimodal/project`，并将其加入 `PYTHONPATH`（如需要）。
2. **三数据集下载与整理**  
   - CREMA-D：运行 `scripts/organize_crema_d.py`，将原始音视频与标签整理到统一目录结构 `data/{train,val,test}/{video,audio,text,labels}`。  
   - MELD：按 MELD 对应脚本（若存在 `organize_meld.py`）或文档说明，将多方对话、音频与文本整理为统一命名的样本。  
   - CMU-MOSEI：  
     - 使用 Kaggle 版 `.csd` 特征，运行 `scripts/organize_cmu_mosei_from_raw.py` 生成中间目录 `data/CMU_MOSEI_MM/...`；  
     - 运行 `scripts/merge_cmu_mosei_to_data.py`，将 MOSEI 样本合并到统一的 `data/{train,val,test}` 目录，与 CREMA-D / MELD 共存。  
3. **目录结构版数据清洗（推荐在所有实验前统一做一次）**  
   - 只读检测：  
     - 运行 `python scripts/check_media_health_dir.py --data_dir data`，生成 `bad_samples_train_dir.csv`、`bad_samples_val_dir.csv`。  
   - 可选归档：  
     - 人工确认后运行 `python scripts/move_bad_media.py --data_dir data`，将坏样本移动到 `data/bad/...`，训练与验证时自动跳过。  
   - 说明：当前 AT/T/A 等基线与 CREMA 微调实验已经基于该清洗策略跑通，后续新实验建议沿用同一清洗版本，保证可比性。

### 9.2 统一配置与日志命名策略（已完成，可沿用）

1. **基础配置文件**  
   - 所有实验以 `config/config.yaml` 为模板，根据需要复制出衍生配置：  
     - `config/config_text_only.yaml`（T-only）、`config/config_audio_only.yaml`（A-only）、`config/config_AT_DA.yaml`（AT+DA）、`config_crema_finetune_from_pretrain.yaml`、`config_crema_finetune_from_scratch.yaml` 等。  
   - 关键结构保持一致：  
     - `model.modalities.use_{audio,video,text,physiological}` 控制模态开关；  
     - `model.domain_adaptation.enabled` 与 `training.loss.use_domain_adaptation` 控制 DA；  
     - `training.mode` 控制 `pretrain` / `finetune`；  
     - `training.loss.{class,reg,trend}_loss_weight` 控制多任务损失权重。
2. **日志与 checkpoint 命名约定**  
   - 预训练：`{modalities}_pretrain_3datasets_{noDA/DA}_YYYYMMDD_HHMMSS`，例如：  
     - `AT_pretrain_3datasets_noDA_20260305`、`T_pretrain_3datasets_noDA_20260311`、`A_pretrain_3datasets_noDA_20260311`、`AT_pretrain_3datasets_DA_20260312_201945`。  
   - 微调：`{modalities}_crema_{from_pretrain/scratch}_{noDA/DA}_YYYYMMDD_HHMMSS`，例如：  
     - `AT_crema_from_pretrain_noDA_20260313_212316`、`AT_crema_scratch_noDA_20260314_102323`。  
   - 最优 checkpoint 统一为：`checkpoints/checkpoint_pretrain_best.pth`（预训练）与 `checkpoints/checkpoint_finetune_best.pth`（如有需要）。

### 9.3 多模态基线与单模态消融（已完成）

> 本节对应 10.5、10.5.1、10.5.2；你已经完成了所有训练与记录。这里以“操作步骤+总结”形式重述，方便今后复现或迁移到其他任务。

1. **AT Baseline（三数据集混合，无 DA）——已完成**  
   - **配置**：  
     - 模态：`use_audio=true, use_text=true, use_video=false, use_physiological=false`；  
     - 域适应：`model.domain_adaptation.enabled=false`，`training.loss.use_domain_adaptation=false`；  
     - 任务：仅分类任务（ClassBalancedLoss），回归与趋势任务权重为 0；  
     - 训练：`batch_size=4`，`num_epochs=50`，`learning_rate=1e-4`。  
   - **运行命令示例**：  
     - `python scripts/train.py --config config/config.yaml --mode pretrain`。  
   - **结果记录**：详见 `10.5`（AT Baseline 的完整指标与日志）。
2. **T-only 文本单模态基线（无 DA）——已完成**  
   - **配置**：  
     - 在 `config_text_only.yaml` 中设定：`use_text=true, use_audio=false, use_video=false`，其余与 AT Baseline 一致。  
   - **运行命令示例**：  
     - `python scripts/train.py --config config/config_text_only.yaml --mode pretrain`。  
   - **结果记录**：详见 `10.5.1`（T-only 的完整指标与日志）。
3. **A-only 音频单模态基线（无 DA）——已完成**  
   - **配置**：  
     - `use_audio=true, use_text=false, use_video=false`；其它与 AT Baseline 一致。  
   - **运行命令示例**：  
     - `python scripts/train.py --config config/config_audio_only.yaml --mode pretrain`。  
   - **结果记录**：详见 `10.5.2`（A-only 的完整指标与日志）。

### 9.4 域适应消融：AT+DA vs AT noDA（已完成）

> 对应 10.5.3，你已经完成 AT+DA 预训练并与 AT Baseline 做了对比。

1. **配置要点**  
   - 在 `config_AT_DA.yaml` 中：  
     - 模态同 AT Baseline：`use_audio=true, use_text=true, use_video=false`；  
     - 域适应：`model.domain_adaptation.enabled=true`，`training.loss.use_domain_adaptation=true`，`training.loss.domain_loss_weight=0.1`；  
     - 训练参数与 AT Baseline 对齐（50 epoch，batch_size=4，lr=1e-4）。  
2. **运行命令示例**  
   - `python scripts/train.py --config config/config_AT_DA.yaml --mode pretrain`。  
3. **结果与结论**（10.5.3）  
   - 详细数值与结论详见 `10.5.3`。

### 9.5 单数据集微调：CREMA-D（已完成 AT 模态）

> 对应 `10.5.4`，你已经完成“预训练+CREMA 微调”与“CREMA scratch”两条实验。

1. **预训练+微调链路（from_pretrain）**  
   - **配置**：`config_crema_finetune_from_pretrain.yaml`：  
     - 模态：`use_audio=true, use_text=true, use_video=false`（与 AT Baseline 一致）；  
     - 训练模式：`mode=finetune`；  
     - 目标数据集：`training.finetune.datasets: [crema]`；  
     - resume：`checkpoints/checkpoint_pretrain_best.pth`（AT Baseline 三数据集预训练最优权重）。  
   - **运行命令示例**：  
     - `python scripts/train.py --config config_crema_finetune_from_pretrain.yaml --mode finetune`。  
   - **结果**：详见 `10.5.4`（AT_crema_from_pretrain_noDA）。
2. **从零训练链路（scratch）**  
   - **配置**：`config_crema_finetune_from_scratch.yaml`：  
     - 模态同上，但不从预训练 checkpoint 恢复（或只加载非预训练权重）；  
     - 训练轮数、lr 与 from_pretrain 对齐（30 epoch）。  
   - **运行命令示例**：  
     - `python scripts/train.py --config config_crema_finetune_from_scratch.yaml --mode finetune`。  
   - **结果**：详见 `10.5.4`（AT_crema_scratch_noDA）。
3. **综合结论（预训练收益）**  
   - 综合结论详见 `10.5.4`，后续在论文中用于“预训练 vs scratch”对比表。

### 9.6 后续扩展：引入视频模态的全流程指导（阶段一已完成，阶段二待做）

在已有 AT/T/A/AT+DA/AT+CREMA 微调的基础上，已将**视频模态**按 **V-only → VT → AVT（无 DA）** 跑通并写入 **第十章 10.5.5–10.5.7**。本节保留操作步骤供复现；**详细数值与结论以 10.5.x 为准**，避免与本节重复堆砌。

#### 9.6.1 阶段一：视频模态基线（无 DA）——已完成

1. **V-only（三数据集混合，video-only）** — **已完成**  
   - **目标**：量化“仅视频”在三数据集混合场景下的可用程度，与 T-only/A-only/AT 对比。  
   - **配置文件**：`config/config_video_only.yaml`。  
   - **运行命令**：`python scripts/train.py --config config/config_video_only.yaml --mode pretrain`。  
   - **结果与日志**：**10.5.5**，run 示例 `logs/V_only_pretrain_3datasets_noDA_20260317_205923`。
2. **VT（Video+Text，无 DA）** — **已完成（含续训合并）**  
   - **配置文件**：`config/config_VT_noDA.yaml`。  
   - **运行命令**：`python scripts/train.py --config config/config_VT_noDA.yaml --mode pretrain`。  
   - **结果与日志**：**10.5.6**（首段 + 续训两段合并为完整 1–50 epoch）。
3. **AVT（Audio+Video+Text，无 DA，全模态）** — **已完成（2026‑03‑23）**  
   - **配置文件**：`config/config_AVT_noDA.yaml`。  
   - **运行命令**：`python scripts/train.py --config config/config_AVT_noDA.yaml --mode pretrain`。  
   - **结果与日志**：**10.5.7**，run `logs/AVT_pretrain_3datasets_noDA_20260323_202809`。  
   - **结果解读**：详见 `10.5.7`（完整指标、曲线形态与论文表述建议）。

#### 9.6.2 阶段二：视频模态 + 域适应（DA）

1. **VT+DA（三数据集混合）**  
   - **目标**：在包含视频的双模态场景下评估 DA 的价值。  
   - **推荐配置文件**：`config/config_VT_DA.yaml`：  
     - 模态同 VT；  
     - `model.domain_adaptation.enabled=true`，`training.loss.use_domain_adaptation=true`，`domain_loss_weight=0.1`。  
   - **运行命令示例**：  
     - `python scripts/train.py --config config/config_VT_DA.yaml --mode pretrain`。  
   - **对比对象**：VT_noDA vs VT_DA；关注 val/accuracy、val/f1 与 val/loss_total 的变化。  
2. **AVT+DA（三模态 + 域适应，可选）**  
   - **目标**：在全模态场景下测试 DA 的效果（资源允许时）。  
   - **推荐配置文件**：`config/config_AVT_DA.yaml`，在 AVT 配置基础上开启 DA。  
   - **运行命令** 与记录方式同 VT+DA。

#### 9.6.3 阶段三：下游单数据集微调（含视频）

1. **优先建议：VT on MELD**  
   - **目标**：在带有对话与视频信息的 MELD 上验证「VT 预训练+微调 vs VT scratch」。  
   - **步骤概述**：  
     1. 选择 VT 或 AVT 的三数据集预训练 checkpoint（如 `VT_pretrain_3datasets_noDA_YYYYMMDD`）。  
     2. 复制 `config_crema_finetune_from_pretrain.yaml`，改名为 `config_meld_VT_finetune_from_pretrain.yaml`，只修改：  
        - 目标数据集为 MELD（如 `training.finetune.datasets: [meld]`）；  
        - 模态开关为 VT 或 AVT；  
        - resume 路径为对应 VT/AVT 预训练 checkpoint。  
     3. 再复制一份 scratch 版本 `config_meld_VT_finetune_from_scratch.yaml`，去掉 resume。  
     4. 分别运行 from_pretrain 与 scratch，比较 MELD 验证集上的 F1/Accuracy。  
   - **记录要求**：在 10.5.4 之后新增 “MELD 单数据集微调（VT/AVT）” 小节，结构与 CREMA-D 微调小节一致。
2. **可选：AVT on CREMA-D / MOSEI**  
   - 复用 CREMA-D 配置，将模态改为 AVT，并分别做 from_pretrain / scratch 对比。  
   - 对于 MOSEI，可根据数据规模调整 epoch 与 batch_size。

### 9.7 AVT 无DA 训练执行 SOP（tmux + TensorBoard，服务器防断连）

本节给出可直接复制执行的完整操作流程，默认服务器已有 Conda 与依赖环境。

#### 9.7.1 启动前检查（建议每次都做）

1. 进入项目目录并激活环境  
   - `cd /home/lizhichun_24/sda1/code/multimodal/project`  
   - `conda activate <你的环境名>`
2. 检查配置文件是否存在  
   - `ls config/config_AVT_noDA.yaml`
3. （推荐）先做数据健康检查，避免坏视频拖慢训练  
   - `python scripts/check_media_health_dir.py --data_dir data`

#### 9.7.2 使用 tmux 启动 AVT 无DA 训练

1. 新建并进入会话  
   - `tmux new -s train_avt_noda`
2. 启动训练  
   - `python3 scripts/train.py --config config/config_AVT_noDA.yaml --mode pretrain`
3. 脱离会话（不中断训练）  
   - `Ctrl + b`，再按 `d`
4. 重新连接会话查看进度  
   - `tmux attach -t train_avt_noda`
5. 结束训练（如需手动停止）  
   - 进入会话后按 `Ctrl + C`

#### 9.7.3 TensorBoard 记录与远程访问（推荐固定 6007 端口）

1. 在服务器开一个独立 tmux 会话跑 TensorBoard  
   - `tmux new -s tb_avt`  
   - `cd /home/lizhichun_24/sda1/code/multimodal/project`  
   - `tensorboard --logdir logs --host 0.0.0.0 --port 6007`
2. 本地机器做 SSH 端口转发（按你的服务器信息）  
   - `ssh -p 1022 -L 6007:localhost:6007 <your_user>@49.233.89.203`
3. 浏览器访问  
   - `http://localhost:6007`
4. 详细的 TensorBoard 指标含义与常用曲线（train/val 的 loss 与 accuracy/f1）见 `7.2 TensorBoard 可视化`。

#### 9.7.4 训练中断恢复（统一规范）

1. 从 checkpoint 恢复  
   - `python3 scripts/train.py --config config/config_AVT_noDA.yaml --mode pretrain --resume checkpoints/checkpoint_pretrain_epoch_XX.pth`
2. 关键说明  
   - 当前 `scripts/train.py` 已恢复 `model + optimizer + scheduler`（pretrain 模式），可避免学习率轨迹重置导致的 TensorBoard 曲线断崖。  
3. 建议  
   - 恢复后第一时间对比学习率与 loss 走势是否连续；必要时记录到周报“异常处理”页。

### 9.8 截至目前“实际使用模型”盘点与后续换模型计划

#### 9.8.1 你目前所有实验的共同模型骨架（统一结论）

- **主模型始终是同一个**：`models/multimodal_model.py` 中的 `MultimodalEmotionModel`。  
- 训练脚本固定调用：`scripts/train.py` 中直接实例化 `MultimodalEmotionModel(config)`。  
- 已完成实验（AT/T/A/V/VT/**AVT**、CREMA 微调）之间的差异，**主要来自配置文件**：  
  1. 模态开关（`use_video/use_audio/use_text`）；  
  2. 是否启用 DA（`model.domain_adaptation.enabled`、`training.loss.use_domain_adaptation`）；  
  3. 数据与训练超参数（batch、视频帧数/分辨率、学习率等）。  

> 结论：到目前为止你不是“换了不同主模型在跑”，而是在同一主模型框架里做了配置驱动的模态/DA消融。

#### 9.8.2 当前是否需要“换模型”？

- **短期建议：不需要立即换主模型骨架**。  
  - 原因：**AVT noDA 全量结果已记入 10.5.7**，当前主线可转向 **VT+DA / AVT+DA**、融合策略与下游微调，仍应保持同一 `MultimodalEmotionModel` 以保证可比；  
  - 若现在换主模型，会破坏与已完成 AT/VT/V-only/AVT 的可比关系。  
- **可以做的“轻量模型切换”**：保持 `MultimodalEmotionModel` 不变，仅切换融合策略配置：  
  - `fusion_strategy: "standard"`（当前稳定基线）  
  - `fusion_strategy: "emotion_shift"`  
  - `fusion_strategy: "leader_follower"`  
  - `fusion_strategy: "two_stage"`

#### 9.8.3 后续模型实验计划（建议顺序）

1. **AVT_noDA 基线** — **已完成**（`logs/AVT_pretrain_3datasets_noDA_20260323_202809`，见 **10.5.7**）。  
2. **做 AVT+DA 对照**（当前优先实验之一）  
   - 目标：验证 DA 在三模态场景是否有效。  
3. **在不换骨架前提下做融合策略消融**  
   - 固定模态（建议先 VT，再 AVT），逐个对比 `standard/emotion_shift/leader_follower/two_stage`。  
4. **若以上都稳定后，再考虑“真正换主干模型”**（可选）  
   - 例如升级视频 backbone、或替换文本编码器；  
   - 每次只改一个主干，并复用同一训练配方做公平对比。

#### 9.8.4 若后续要换融合策略，最小改动步骤

1. 复制当前配置（如 `config_AVT_noDA.yaml`）为新文件（例如 `config_AVT_noDA_emotion_shift.yaml`）；  
2. 只改 `model.attention.fusion_strategy`；其余超参数不变；  
3. 运行同样命令开始训练；  
4. 在 10.5 新增小节记录“同模态、不同融合策略”的结果与结论。  

### 9.9 AVT 异常定位最小实验矩阵（针对 noDA 中期阶段性异常；全程 50 epoch 已结束）

> **状态**：同一 run（`AVT_pretrain_3datasets_noDA_20260323_202809`）已跑满 **50 epoch**，完整指标与结论见 **10.5.7**；本节矩阵仍可用于 **后续短跑定位**（换配方前对照）。  
> **历史现象**：中期（约 26 epoch 前后）曾出现“train loss 不降反升、val loss 偏高且波动、部分 epoch 上 `precision≈1.0`”等形态。  
> **目标**：用最少对照实验判断问题是否主要来自 **学习率/损失配置/DA缺失/融合策略**。

#### 9.9.1 执行原则（保证论文可比性）

1. **先保留当前 AVT_noDA 全程到 50 epoch**，作为“异常基线”完整证据，不中断、不覆盖；  
2. 后续定位实验全部采用“**单变量改动**”原则；  
3. 短跑阶段统一设为 `num_epochs=15`（用于快速定位，不作为最终主结果）；  
4. 统一记录：`Best/Last Acc/F1`、`val/loss_total` 区间、是否出现 `precision=1.0` 固定现象。

#### 9.9.2 最小矩阵（4 组短跑，按顺序执行）

| 组别 | 目的 | 配置文件建议 | 仅改动项 | 运行命令 |
|---|---|---|---|---|
| S0（参考） | 保留异常现象全程证据 | `config/config_AVT_noDA.yaml` | 无（继续当前训练到 50 epoch） | `python3 scripts/train.py --config config/config_AVT_noDA.yaml --mode pretrain --resume checkpoints/checkpoint_pretrain_epoch_XX.pth` |
| S1（LR定位） | 判断是否由学习率偏大导致训练抖动/退化 | `config/config_AVT_noDA_diag_lr5e5.yaml` | `training.learning_rate: 5e-5`，`num_epochs: 15` | `python3 scripts/train.py --config config/config_AVT_noDA_diag_lr5e5.yaml --mode pretrain` |
| S2（损失定位） | 判断 ClassBalancedLoss 是否导致预测塌缩 | `config/config_AVT_noDA_diag_ce.yaml` | `training.loss.use_class_balanced: false`，其余不变，`num_epochs: 15` | `python3 scripts/train.py --config config/config_AVT_noDA_diag_ce.yaml --mode pretrain` |
| S3（DA定位） | 判断 DA 在 AVT 上是否缓解异常 | `config/config_AVT_DA_diag.yaml` | 使用 AVT_DA 配方并将 `num_epochs: 15` | `python3 scripts/train.py --config config/config_AVT_DA_diag.yaml --mode pretrain` |
| S4（融合定位） | 判断 `emotion_shift` 是否优于 standard | `config/config_AVT_noDA_emotion_shift_diag.yaml` | 使用 emotion_shift 配方并将 `num_epochs: 15` | `python3 scripts/train.py --config config/config_AVT_noDA_emotion_shift_diag.yaml --mode pretrain` |

#### 9.9.3 结果判定规则（短跑后立即判断）

1. **优先判定“是否脱离异常态”**：  
   - 若 `precision=1.0` 固定现象明显减少；  
   - 且 `val/loss_total` 中位数显著下降（相对 S0 中期）；  
   - 且 `Best F1` 与 `Last F1` 差距缩小（后期不再明显回落）。  
2. **若 S1 有明显改善**：后续主线优先采用较小学习率；  
3. **若 S2 有明显改善**：说明类别平衡损失在当前 AVT 设置下可能过强，后续改为 CE 或再调 `beta`；  
4. **若 S3 有明显改善**：优先推进 AVT_DA 全程 50 epoch；  
5. **若 S4 有明显改善**：在 noDA 与 DA 两条线上都增加 emotion_shift 对照。

#### 9.9.4 论文记录口径（可直接写进实验章）

- 本轮短跑为“异常定位实验”，目的在于识别导致 AVT_noDA 中期异常的主导因素；  
- 其结果用于确定后续 50 epoch 正式对照配方，不与主结果表直接并列；  
- 主结果表仅收录完整训练（50 epoch）实验，短跑结果放在“异常分析/补充实验”小节。

#### 9.9.5 推荐执行命令（tmux）

1. `tmux new -s avt_diag`  
2. `cd /home/lizhichun_24/sda1/code/multimodal/project`  
3. 按 S1 -> S2 -> S3 -> S4 顺序运行  
4. 每组完成后记录到 `docs/PROJECT_OVERVIEW_AND_TRAINING_PLAN.md` 第 10 章实验日志  
5. `Ctrl+b, d` 脱离会话，防止断连中断

### 9.10 论文主线重跑最小闭环清单（最终执行版）

> 目的：在“指标口径已修复（precision/recall/f1 计算修正）”前提下，以最小成本重建论文核心证据链。  
> 原则：只重跑论文主结论需要的实验，不重跑全部历史探索实验。

#### 9.10.1 必须重跑的实验（按优先级）

1. **AT_noDA（主基线）**  
   - 作用：所有对照组的统一锚点，论文主表必需。  
2. **AT_DA（DA 对照）**  
   - 作用：完成 AT 的 noDA vs DA 最小闭环。  
3. **VT_noDA（视频引入关键中间态）**  
   - 作用：回答“引入视频后是否有收益”的关键证据。  
4. **AVT_noDA（三模态主线）**  
   - 作用：全模态主结果，论文核心。  
   - **历史 run**：`logs/AVT_pretrain_3datasets_noDA_20260323_202809` 已完整记录于 **10.5.7**；若在「指标计算修复」后需与 AT/VT 完全同口径，可按同一配置 **选择性重跑** 或 **`recompute_val_metrics.py` 复核**。  
5. **AVT_DA（三模态 DA 对照）**  
   - 作用：回答 DA 在 AVT 上是否稳定有效。  
6. **AVT_noDA_emotion_shift（融合策略对照）**  
   - 作用：`standard` vs `emotion_shift`，支撑方法章结构消融结论。

#### 9.10.2 可保留历史结果、不必重跑

- `T_only` / `A_only` / `V_only`：可作为补充背景结果引用；  
- 若时间紧张，不作为本轮重跑主线，以降低总训练成本。

#### 9.10.3 若时间允许的加分项（建议补 1 组）

- **MELD：from_pretrain vs scratch（VT 或 AVT 任选一组）**  
  - 作用：增强“迁移与泛化”章节说服力，避免仅 CREMA 单点结论。

#### 9.10.4 推荐执行顺序（严格按此顺序）

1. `AT_noDA`  
2. `AT_DA`  
3. `VT_noDA`  
4. `AVT_noDA`  
5. `AVT_DA`  
6. `AVT_noDA_emotion_shift`  
7. （可选）MELD 迁移对照

#### 9.10.5 逐条可复制命令清单（tmux 版）

> 统一前置命令（每个会话先执行）：
>
> - `cd /home/lizhichun_24/sda1/code/multimodal/project`  
> - `conda activate <你的环境名>`

**[1] AT_noDA**

```bash
tmux new -s train_at_noda
cd /home/lizhichun_24/sda1/code/multimodal/project
conda activate <你的环境名>
python3 scripts/train.py --config config/config.yaml --mode pretrain
```

**[2] AT_DA**

```bash
tmux new -s train_at_da
cd /home/lizhichun_24/sda1/code/multimodal/project
conda activate <你的环境名>
python3 scripts/train.py --config config/config_AT_DA.yaml --mode pretrain
```

**[3] VT_noDA**

```bash
tmux new -s train_vt_noda_mainline
cd /home/lizhichun_24/sda1/code/multimodal/project
conda activate <你的环境名>
python3 scripts/train.py --config config/config_VT_noDA.yaml --mode pretrain
```

**[4] AVT_noDA**

```bash
tmux new -s train_avt_noda_mainline
cd /home/lizhichun_24/sda1/code/multimodal/project
conda activate <你的环境名>
python3 scripts/train.py --config config/config_AVT_noDA.yaml --mode pretrain
```

**[5] AVT_DA**

```bash
tmux new -s train_avt_da_mainline
cd /home/lizhichun_24/sda1/code/multimodal/project
conda activate <你的环境名>
python3 scripts/train.py --config config/config_AVT_DA.yaml --mode pretrain
```

**[6] AVT_noDA_emotion_shift**

```bash
tmux new -s train_avt_noda_es_mainline
cd /home/lizhichun_24/sda1/code/multimodal/project
conda activate <你的环境名>
python3 scripts/train.py --config config/config_AVT_noDA_emotion_shift.yaml --mode pretrain
```

**通用 tmux 操作**

- 脱离会话：`Ctrl+b`，再按 `d`  
- 回到会话：`tmux attach -t <session_name>`  
- 查看会话列表：`tmux ls`

#### 9.10.6 TensorBoard 监控（可复制）

```bash
tmux new -s tb_mainline
cd /home/lizhichun_24/sda1/code/multimodal/project
conda activate <你的环境名>
tensorboard --logdir logs --host 0.0.0.0 --port 6007
```

本地转发：

```bash
ssh -p 1022 -L 6007:localhost:6007 <your_user>@49.233.89.203
```

浏览器访问：`http://localhost:6007`

#### 9.10.7 每个实验结束后必须记录的字段（论文防返工）

1. run 名称；  
2. config 路径；  
3. `Best Acc/F1`；  
4. `Last Acc/F1`；  
5. 总时长；  
6. TensorBoard 关键截图（train/loss_classification、val/f1）；  
7. 1-2 句结论（是否优于对照、是否稳定）。

#### 9.10.8 执行后的写作映射（章节对应）

- `AT_noDA vs AT_DA` -> 论文“DA 消融”主结论；  
- `VT_noDA` -> 论文“视频引入增益”中间证据；  
- `AVT_noDA vs AVT_DA` -> 论文“全模态 + DA”主结论；  
- `AVT_noDA_standard vs AVT_noDA_emotion_shift` -> 论文“融合策略消融”主结论。

---

通过以上 9.1–9.8 的步骤，你可以从**环境准备 → 三数据集预训练 → 模态/DA 消融 → 单数据集微调 → 视频模态扩展 → 远程训练与模型演进规划**完整走一遍实验流程，并且在需要复现实验或迁移到新任务时，只需复用本节的步骤与配置模板即可。


## 十、如何利用日志做论文分析（从 CSV 到论文图表）

1. **收敛与稳定性**：
   - 从 `metrics.csv` 中提取 `loss`、`accuracy` 等列，用 Matplotlib/Seaborn 绘制 epoch vs 值的曲线；
   - 对比不同实验（不同配置）下曲线的收敛速度与最终性能。

2. **类别不平衡与域适应效果**：
   - 对比：
     - `use_class_balanced=true` vs `false`；
     - `use_domain_adaptation=true` vs `false`；
   - 重点关注少数类别 F1、整体 F1 和泛化性能（例如在某个单一数据集上测试）。

3. **不同数据集贡献**：
   - 通过 BalancedSampler 的统计结果和各数据集单独测试集上的指标，分析混合训练对单一数据集性能的影响；
   - 对比仅在单数据集上训练 vs 混合训练 + 微调。

4. **训练稳定性与超参数敏感性**：
   - 对比不同学习率、batch size、loss 权重配置下的训练曲线，观察是否出现震荡、过拟合等现象。

---

## 十一、实验记录与工作日志（混合数据集预训练，重要）

> 本小节是你在 Cursor 中实际调试过程的文字记录，方便日后回顾与论文写作。  
> 建议后续每次做关键改动时，都在这里追加一两行说明。

> **小节编号说明**：为与历史笔记、脚本及 TensorBoard run 注释对齐，本章仍使用 **`10.0`–`10.6`、`10.5.x`** 作为小节标题；在全书章节编号中，本章对应 **第十一章**。

### 10.0 本章导航（结果索引，便于检索）

| 小节 | 内容 | 状态 |
|------|------|------|
| 10.1–10.4 | 调试问题、Baseline 选型、路线建议、客观限制 | 背景 |
| **10.5** | **AT Baseline（三数据集、无 DA）** | 已完成 |
| **10.5.1 / 10.5.2** | **T-only / A-only** | 已完成 |
| **10.5.3** | **AT+DA** | 已完成 |
| **10.5.4** | **CREMA 微调：预训练 vs scratch** | 已完成 |
| **10.5.5** | **V-only（无 DA）** | 已完成 |
| **10.5.6** | **周日志 + VT 无 DA（含续训合并）** | 已完成 |
| **10.5.7** | **AVT 三模态无 DA（全量 50 epoch）** | **已完成（2026‑03‑23）** |
| **10.6** | AT 之后的通用「下一步」操作指引（复现、T/A 消融、微调思路） | 长期参考 |

**写作约定**：同一类结论优先只在 **10.5.x 对应小节** 写完整数字；第九章「9.3–9.6」为流程重述，与第十章交叉引用，避免两处各写一套互相打架。

### 10.1 初次跑 VAT + 域适应 + EmotionShift 时遇到的问题

- **初始目标**：直接按照“全模态 VAT + 域适应 + EmotionShift 融合”的配置，完成 CREMA‑D + MELD + MOSEI 的统一预训练。
- **主要问题与解决路径（按时间线简要记录）：**
  1. **数据加载阶段**：  
     - 问题：`audio_path` / `video_path` 等字段有缺失，`os.path.exists(None)` 导致 `TypeError`。  
     - 解决：在 `MultimodalDataset._load_*` 函数中统一对 `None` 做判空处理，缺失文件直接返回 `None`。
  2. **BalancedDatasetSampler 初始化过慢 / 卡住**：  
     - 问题：采样器最初通过 `dataset[idx]` 遍历所有样本，导致初始化阶段就尝试读取全部视频/音频，速度极慢且打印大量 ffmpeg 警告。  
     - 解决：改为直接遍历 `dataset.data_list` 中的元信息建立索引，避免在采样器初始化阶段加载模态数据。
  3. **HuggingFace 模型加载错误**：  
     - 问题：音频 backbone 名称写成 `wav2vec2-base`，HuggingFace 仓库实际为 `facebook/wav2vec2-base`。  
     - 解决：在 `config.yaml` 中修正 `model.audio.backbone` 配置。
  4. **模型配置顺序 bug**：  
     - 问题：`MultimodalEmotionModel` 中在定义 `output_config` 之前就引用了它，导致 `UnboundLocalError`。  
     - 解决：将 `output_config = model_config['output']` 和 `hidden_dim = attention_config['hidden_dim']` 提前，统一在多个模块中复用。
  5. **DataLoader collate 报错 `batch must contain tensors ... NoneType`**：  
     - 问题：部分样本缺失某些模态，`__getitem__` 返回的字典中字段为 `None`，`default_collate` 无法拼接。  
     - 解决：在 `__getitem__` 中为缺失模态创建全零占位张量（视频 / 音频 / 生理 / 文本），保证 DataLoader 始终拿到 tensor。
  6. **显存不足（OOM）问题**：  
     - 问题：初始设置为 `batch_size=16`、`frame_size=224`、`num_frames=16`，多模态 + 情感转变模型在 24GB 显存上反复 OOM。  
     - 解决（多次迭代）：  
       - 将 `batch_size` 逐步降到 **4**；  
       - 将 `data.video.frame_size` 从 224 降到 **160**；  
       - 将 `data.video.num_frames` 从 16 降到 **8**。
  7. **域适应模块与 EmotionShift 融合引发的 autograd 错误**：  
     - 多次出现 `Trying to backward through the graph a second time`、`inplace operation` 等错误，主要来自：  
       - 域对抗模块（`DomainAdversarialModule` + GRL）；  
       - EmotionShift 时序融合在 3D 特征上的使用；  
       - 与外层损失计算的交互。  
     - 解决策略：  
       - **第一步**：统一将 EmotionShift 输出的 `(B,T,C)` logits 在时间维上做平均池化，变为 `(B,C)`；  
       - **第二步**：将域分类器的输入从时序特征改为池化后的 `(B,hidden_dim)`；  
       - **第三步（关键取舍）**：为先保证训练稳定，暂时 **关闭域适应模块**（`model.domain_adaptation.enabled=false`，`training.loss.use_domain_adaptation=false`），并将融合策略改为 `standard`，把复杂模块留到后续消融实验中单独开启。
  8. **数据集特定归一化统计更新引发的 VarBackward0 异常**：  
     - 在开启 `torch.autograd.set_detect_anomaly(True)` 之后，新的异常指向 `DatasetSpecificNormalization.update_statistics` 内部对 `features.var()` 的调用（`VarBackward0` 返回 `nan`），说明我们在统计更新时误把 `fused_features` 的计算图也“拖进来了”；  
     - 解决方式：在 `MultimodalEmotionModel.forward` 中调用 `update_statistics` 时对输入做显式 `.detach()`，使统计更新仅作为数值累计过程，不再参与梯度计算：  
       - `self.dataset_norm.update_statistics(fused_features.detach(), dataset_ids.detach())`  
     - 该修改已在代码中生效，并同步记录在此，后续如有需要，可以把这一点写入论文的“工程优化与稳定性处理”小节。
  9. **回归损失数值不稳定导致 MseLossBackward0 返回 NaN**：  
     - 问题：在 AT 基线下继续训练时，`train_epoch` 中的总 loss 出现 `loss=inf`，`detect_anomaly` 报告异常来源于回归分支的 `MseLossBackward0`；  
     - 分析：情绪维度回归（valence/arousal）当前更多是辅助任务，且标签来源/范围可能存在噪声或极端值，直接参与训练容易导致数值不稳定；  
     - 临时解决策略：在当前阶段**仅优化分类性能**，将配置中的  
       - `training.loss_weights.regression` 和 `trend` 设为 `0.0`，并在 `MultimodalLoss` 中根据权重是否大于 0 决定是否计算对应的 MSELoss；  
       - 这样既保留了未来重新启用回归/趋势预测的接口，又可以在现有环境下先稳定跑通分类任务的混合预训练。
  10. **分类损失中 LogSoftmaxBackward0 返回 NaN（logits 数值爆炸）**：  
      - 问题：继续训练 AT Baseline 时，`ClassBalancedLoss` 的 `classification` 项在若干 batch 后出现 `1e36` 量级，`detect_anomaly` 指向交叉熵内部的 `LogSoftmaxBackward0`，说明部分 `emotion_logits` 中已经出现 NaN/Inf；  
      - 处理策略：在 `MultimodalLoss.forward` 中对 `outputs['emotion_logits']` 进行一次集中数值清洗：  
        - 使用 `torch.nan_to_num` 将 NaN 映射为 0，将正/负无穷截断到有限范围（例如 ±1e4），再送入交叉熵；  
        - 这样可以防止个别 batch 的异常值直接把整条训练曲线“打爆”，同时不影响大多数正常样本的梯度计算；  
      - 同时在 `MultimodalEmotionModel.forward` 中对聚合后的特征 `pooled_features` 做 `nan_to_num` 处理，避免全连接分类头在含有 NaN/Inf 的输入上产生不稳定梯度；  
      - 这一系列数值防护逻辑已写入 `scripts/train.py` 与 `models/multimodal_model.py`，并在本日志中记录，后续在更稳定的环境或更成熟的数据清洗方案下，可以视情况放宽或移除这些截断。

### 10.2 为什么当前 Baseline 先选 “AT 模态 + 无域适应 + 标准注意力”

- 在上述调试过程中，最大的工程难点集中在：  
  - 高分辨率多帧视频通过 ResNet50 产生的显存压力；  
  - 域对抗 + EmotionShift 等高级模块带来的复杂计算图。  
- 考虑到你需要 **先有一个稳定、可复现的混合数据集预训练基线**，再去做各种“+ 模块”的消融实验，因此当前做了如下折中：
  1. **先关闭视频模态（`use_video=false`）**，以“**音频+文本 AT 组合**”作为统一情感建模的第一个 Baseline；  
  2. **关闭域适应模块**，仅保留类别平衡损失 + 数据集特定归一化；  
  3. **融合策略使用标准多头注意力（`fusion_strategy="standard"`）**，EmotionShift / LeaderFollower / TwoStage 暂时不用；  
  4. 调整 batch 大小与视频预处理参数以确保训练不再 OOM。
- 这样得到的 Baseline 更容易收敛，也便于在论文中作为“最基础的跨数据集统一建模框架”，后续所有增强（加回 V 模态、加入 DA、切换融合策略）都可以在此之上逐项对比。

### 10.3 后续实验路线（结合本日志的建议）

1. **阶段 1：AT Baseline 完整跑通**  
   - 按 7.1.1 中当前配置，先跑完若干 epoch，观察 loss 与指标曲线；  
   - 保存验证集表现最好的 checkpoint，作为后续微调与对比的起点。
2. **阶段 2：加回视频模态做模态消融**  
   - 先尝试 `T`、`A`、`AT` 组合跑通；  
   - 在显存和稳定性允许的前提下，再逐步尝试 `VT`、`VAT`，必要时可考虑将视频改为“预提取特征 .npy”而非端到端 ResNet。
3. **阶段 3：开启域适应模块做有/无对比**  
   - 在一个已经稳定的模态组合上（例如 AT 或 VAT），将 DA 开关从 `false` 改为 `true`，复现实验 B；  
   - 对比验证集 F1/Accuracy 与收敛情况。
4. **阶段 4：比较不同融合策略**  
   - 在结构与损失都确定后，切换 `fusion_strategy` 为 `emotion_shift`、`leader_follower`、`two_stage` 等，构成完整的融合策略消融表。

> 建议：今后每次做完一轮关键实验（尤其是改变模态组合 / 域适应 / 融合策略 / 损失设计时），都在本“第十章：工作日志”中加一段几行文字，记录时间、配置和主要结论，相当于论文的“实验日志原始材料”。

### 10.4 当前仍存在的困难与客观限制（2026‑03‑05 状态）

- **显存与并发进程限制**：
  - 当前服务器上同一块 RTX 4090 同时有多个进程占用显存（约 6–7GB），本项目在 VAT + DA + EmotionShift 配置下很容易触发 OOM；
  - 为避免对他人任务造成影响，暂未采用“独占 GPU / 杀掉其他进程”的激进方案，而是通过减小 `batch_size`、分辨率和帧数来规避；
  - 后续如需完整 VAT + DA + EmotionShift 配置，建议在**显存更大的 GPU（≥48GB）或独占环境**下重新尝试，或者在离线预提取视频特征后再训练。

- **复杂模块组合下的 autograd 稳定性**：
  - 在同时启用：ResNet50 视频分支 + EmotionShift 时序融合 + 域对抗模块时，多次出现 autograd 相关错误（second backward / inplace modification 等）；
  - 这些问题通常需要对每个子模块逐一做更细致的梯度检查与简化（例如去掉部分 checkpoint、避免 in-place 操作、拆分计算图），调试周期较长；
  - 鉴于当前论文时间节点，暂时采用“自上而下关掉复杂度”的策略（先关 DA，再关 EmotionShift，最后临时关掉视频模态），优先保证**有一条稳定的混合数据集训练链路**。

- **后续计划中的结构优化方向（非必须，但可列入“未来工作”）**：
  1. **视频特征预提取方案**：  
     - 使用独立脚本或离线流程，预先将视频段转为固定长度的视觉特征（如 OpenFace2 或 ResNet 平均池化后的向量），在训练时只读 `.npy`，彻底避免端到端 ResNet 引起的 OOM 与梯度复杂度。
  2. **模块级别的轻量化**：  
     - 在启用 EmotionShift / TwoStageFusion 时，将 `hidden_dim`、`num_layers` 适当减小，或者只在高层融合后加入一层轻量化的情感转变模块；
  3. **更细粒度的梯度调试**：  
     - 在有余力时，可针对 VAT + DA + EmotionShift 组合使用 `torch.autograd.set_detect_anomaly(True)` 单步定位具体触发 inplace/second-backward 的算子（当前已在 `scripts/train.py` 中开启该选项），并在论文的“未来工作”中说明这是下一步要解决的工程问题。

### 10.5 AT Baseline 首次完整预训练结果（2026‑03‑06）

本节记录 2026‑03‑05 晚至 2026‑03‑06 早完成的 **AT Baseline 三数据集混合预训练**，对应运行目录：  
`logs/multimodal_emotion_recognition_20260305_234413`。

- **实验配置概要（与 7.1.1 对应）**  
  - 模态：`use_audio=true`，`use_text=true`，`use_video=false`，暂不启用生理信号；  
  - 融合：`fusion_strategy="standard"`，关闭 EmotionShift 与两阶段融合；  
  - 域适应 / 归一化：`use_domain_adaptation=false`，`dataset_normalization.enabled=false`；  
  - 损失：仅保留分类任务的 `ClassBalancedLoss`，回归与趋势预测的损失权重均设为 0；  
  - 训练参数：`batch_size=4`，`frame_size=160`，`num_frames=8`，`learning_rate=1e-4`，共 50 个 epoch。

- **训练收敛情况（TensorBoard `train/loss_classification` & `train/loss_total`）**  
  - 起始阶段（Epoch 0）训练总损失约为 **2.49 左右**；  
  - 随着 epoch 推进，训练损失缓慢下降到 **约 2.25 附近**，曲线整体平滑，无数值爆炸或梯度异常；  
  - 对应截图中的绿色曲线可作为论文中“训练损失随 epoch 变化”的折线图或平滑曲线使用。

- **验证集指标整体范围（综合所有 epoch）**  
  - `val/accuracy` 大致位于 **0.08–0.20** 区间波动；  
  - `val/f1` 大致位于 **0.17–0.26** 区间（个别 epoch 接近 0.26）；  
  - 验证损失 `val/loss_total` 多数在 **3.5–5.5** 之间，存在一定抖动，反映出三数据集混合与类别不平衡带来的困难。

- **代表性最优 epoch（按验证集分类性能选取）**  
  - 例如 **Epoch 35**：  
    - 验证集：Loss ≈ **4.00**，Accuracy ≈ **0.197**，Precision ≈ **0.603**，Recall ≈ **0.197**，F1 ≈ **0.256**；  
  - 这些数值可以在论文中作为“AT Baseline 在三数据集混合验证集上的整体性能”，放入 **基线结果表（见 9.5）** 中；  
  - 训练过程中脚本会将当前最优模型权重保存在 `checkpoints/checkpoint_pretrain_best.pth`，建议论文中明确说明“后续微调与消融实验均以该 checkpoint 作为初始化”。

- **稳定性与异常情况说明**  
  - 本次完整 50 epoch 训练过程中，之前反复出现的 `RuntimeError: DivBackward0 / AddmmBackward0 returned NaN` 等问题均未再出现；  
  - 日志中多次出现的 `[mov,mp4,m4a,...] moov atom not found` 为 **ffmpeg 对部分损坏/非标准视频片段的警告**，但不会导致训练中断；在当前 `use_video=false` 的基线设定下，其实际影响较小，可在论文中简要注明为“数据清洗遗留噪声”。  

- **简要效果点评（供后续消融对比）**  
  - 在仅使用 AT 模态、关闭域适应与回归/趋势任务的简化配置下，模型已经能够在三数据集混合验证集上达到 **约 0.19–0.20 的 Accuracy 与 0.25 左右的 F1**，说明预训练确实学到了一定的判别能力；  
  - 但相比于单数据集专门训练或引入视频模态/域适应的理想情况，这一指标仍处于“中等偏低”的水平，主要受跨数据集域差异与类别极不平衡的限制；  
  - 因此，本次 AT Baseline 更重要的意义在于提供了一条 **数值稳定、可完整跑通的工程基线**，后续所有模态消融、域适应与融合策略实验都可以在此基础上进行对比与量化改进。

- **论文写作时如何使用这些数据（建议）**  
  1. **曲线图**：  
     - 选取本次运行的 `metrics.csv` 或 TensorBoard 标量，绘制两类曲线：  
       - 训练/验证总损失随 epoch 变化曲线（`train/loss_total` vs `val/loss_total`）；  
       - 验证集 Accuracy 与 F1 随 epoch 变化曲线（可放在同一图中对比）。  
  2. **结果表**：  
     - 在 9.5 的基线结果表中，取若干关键 epoch（如最佳 F1 的 Epoch 35，以及早期/中期的 5、20、30 epoch）填入 Accuracy/F1，用于说明模型在长时间预训练下的收益有限但稳定；  
  3. **文字分析要点**：  
     - 指出在仅使用 AT 模态、关闭域适应与回归任务的简化配置下，模型已经能在三数据集混合验证集上达到约 **0.19–0.20 的 Accuracy / 0.25 左右的 F1**；  
     - 分析性能受限的可能原因（数据集间域差异大、类别极度不平衡、部分视频/语音质量较差等），并为后续开启视频模态、域适应、或更复杂融合策略的实验提供动机；  
     - 将本次 AT Baseline 作为后续所有改进方法的 **统一对比基线**，保证论文中结果的一致性与可解释性。

#### 10.5.1 多模态音视频数据清洗与坏样本过滤（目录结构版）

在 AT Baseline 以及后续模态消融实验中，训练数据主要采用 `data/{train,val,test}/{video,audio,physiological,text,labels}` 的目录结构组织。为避免损坏或封装异常的音视频片段对训练稳定性和统计结果造成影响，本项目基于目录结构实现了一套“**只读检测 + 可选移动归档**”的数据清洗流程，具体如下：

- **检测脚本：`scripts/check_media_health_dir.py`（只读版）**  
  - 使用方式：  
    - 在项目根目录执行：  
      - `python scripts/check_media_health_dir.py --data_dir data`  
    - 脚本会基于文件名提取 `sample_id`，分别扫描：  
      - `data/train/video/` 与 `data/train/audio/`；  
      - `data/val/video/` 与 `data/val/audio/`。  
  - 健康性判定规则：  
    - 视频：  
      - 若 `cv2.VideoCapture(path).isOpened()` 失败（典型如 ffmpeg 底层 `moov atom not found` 等错误），视为“无法打开”；  
      - 若能打开但 `read()` 无法解码出任意一帧，视为“无有效帧”；  
      - 上述任一情况均将该 `sample_id` 标记为“视频坏样本”。  
    - 音频：  
      - 使用 `librosa.load(path, sr=16000, duration=1.0)` 读取前 1 秒内容，若抛出异常（格式错误、封装损坏等），将该 `sample_id` 标记为“音频坏样本”。  
    - 若同一样本在视频或音频任一模态上判定为坏样本，则整体视为“坏样本”。  
  - 输出结果：  
    - 在 `data/` 目录下生成：  
      - `bad_samples_train_dir.csv`  
      - `bad_samples_val_dir.csv`  
    - 每条记录包含：`split, sample_id, video_path, audio_path, bad_reason`，其中 `bad_reason` 记录具体失败原因（如 `video_cannot_open`、`video_no_frame_decoded`、`audio_exception:...` 等），便于后续人工核查或复现论文中的数据清洗步骤。

- **可选归档脚本：`scripts/move_bad_media.py`（物理移动坏文件）**  
  - 在人工确认 `bad_samples_*_dir.csv` 列表合理后，可选择执行：  
    - `python scripts/move_bad_media.py --data_dir data`  
  - 行为说明：  
    - 逐条读取 `bad_samples_train_dir.csv` / `bad_samples_val_dir.csv`，将其中列出的 `video_path` 与 `audio_path` 移动到：  
      - `data/bad/train/video/` 与 `data/bad/train/audio/`；  
      - `data/bad/val/video/` 与 `data/bad/val/audio/`。  
    - 原始 `data/train/video`、`data/train/audio`、`data/val/video`、`data/val/audio` 目录中不再包含这些坏样本文件；  
    - `move` 而非 `delete`：如需恢复，可从 `data/bad/...` 再移回原目录。  
  - 对训练流程的影响：  
    - `project/data/dataset.py` 的目录结构加载逻辑会自然**只遍历当前目录下的有效文件**，因此在坏样本被移动到 `data/bad/...` 后，后续所有训练与验证过程都会自动跳过这些样本，无需修改配置文件或训练脚本。  
    - 为保证实验间可比性，建议在开始 AT Baseline / 单模态消融 / 后续微调实验前，统一完成一次目录结构健康检查与坏样本归档，使所有实验均在同一套“清洗后数据子集”上进行。

- **论文复现与方法说明建议**  
  - 在论文“数据预处理 / 数据清洗”小节中，可以简要描述：  
    - 数据采用目录结构组织，通过 OpenCV + Librosa 对所有音视频样本进行统一健康检查；  
    - 将无法正常解码的视频（如 ffmpeg 报 `moov atom not found`）和无法加载的音频统一归档至 `data/bad/...`，不参与训练与评估；  
    - 复现者在获得原始开放数据集后，只需运行 `check_media_health_dir.py` 与 `move_bad_media.py`，即可得到与本文实验一致的“清洗后数据子集”。  
  - 这样可以保证：  
    - 清洗过程透明且可复现；  
    - 实验结果不会受少量损坏样本的随机分布影响；  
    - 不同实验（AT Baseline、T-only、A-only、微调等）之间共享同一套干净数据基础。

#### 10.5.1 文本单模态基线（T-only）预训练结果（2026‑03‑11）

本小节记录 2026‑03‑11 早上在 **文本单模态配置（T-only）** 下完成的三数据集混合预训练结果，对应运行目录：  
`logs/multimodal_emotion_recognition_20260311_072235`。

- **实验配置概要（与 10.5 的 AT Baseline 对应）**  
  - 模态：`use_text=true`，`use_audio=false`，`use_video=false`，不使用生理信号；  
  - 其它模型与训练超参数（注意力结构、损失配置、`batch_size=4`、`num_epochs=50`、`learning_rate=1e-4` 等）与 10.5 中的 AT Baseline 保持一致，仅关闭音频模态；  
  - 域适应与数据集特定归一化依旧关闭：`use_domain_adaptation=false`，`dataset_normalization.enabled=false`；  
  - 训练脚本通过 `tmux` 会话运行完整 50 个 epoch，避免中途 SSH 断开导致训练中断。

- **训练收敛情况（TensorBoard `train/loss_classification` & `train/loss_total`）**  
  - 初始阶段（Epoch 0）训练分类损失约为 **2.56 左右**，略高于 AT Baseline 的 2.49；  
  - 随着 epoch 增加，训练损失整体单调下降，在 Epoch 49 时收敛到 **约 2.11 附近**，趋势平滑稳定，无数值爆炸或异常震荡；  
  - 与 AT Baseline 的曲线对比可以看出，T-only 在长时间预训练下同样能够保持稳定下降，但最终损失略高于 AT 的联合模态训练。

- **验证集指标整体范围（综合所有 epoch）**  
  - `val/accuracy` 主要分布在 **约 0.04–0.22** 区间，其中多数 epoch 位于 **0.12–0.20**；  
  - `val/f1` 大致位于 **约 0.14–0.27** 之间，峰值略高于部分 AT Baseline epoch，但整体波动也更明显；  
  - 验证总损失 `val/loss_total` 介于 **约 4.0–4.6**，整体水平略高于 AT Baseline，反映出仅使用文本模态在三数据集混合情形下的拟合难度。

- **代表性最优 epoch（按验证集 F1 选取）**  
  - 在完整 50 个 epoch 中，`val/f1` 在多个 epoch（如 20、28、39–41 等）达到 **约 0.17–0.20** 的较优水平：  
    - 例如 Epoch 28：Loss ≈ **4.02**，Accuracy ≈ **0.157**，F1 ≈ **0.171**；  
    - Epoch 39–41 的 F1 稳定在 **约 0.17–0.18**，说明模型在后期仍保持一定的泛化能力。  
  - 该运行的详细指标已记录在 `logs/multimodal_emotion_recognition_20260311_072235/metrics.csv`，后续论文撰写时可在“模态消融实验”表格中选取若干代表 epoch（如 5、20、35、50）进行对比。

- **与 AT Baseline 的对比与分析要点**  
  - 在完全相同的训练策略下，T-only 预训练能够在验证集上达到约 **0.16–0.20 的 Accuracy 与 0.18–0.20 左右的 F1**，整体略低于 AT Baseline（约 0.19–0.20 Acc / 0.25 F1）；  
  - 说明在三数据集混合场景中，**单独文本信息不足以完全弥补语音模态的缺失**，音频在情绪识别中仍提供了重要的补充线索；  
  - 本次 T-only 实验将作为后续 A-only、AT 以及引入视频/域适应等方法的对照基线，用于在“模态消融实验”小节中展示不同模态组合对整体性能的定量贡献。

#### 10.5.2 音频单模态基线（A-only）预训练结果（2026‑03‑11）

本小节记录 2026‑03‑11 晚上在 **音频单模态配置（A-only）** 下完成的三数据集混合预训练结果，对应运行目录：  
`logs/multimodal_emotion_recognition_20260311_182615`。

- **实验配置概要（与 10.5 的 AT Baseline / 10.5.1 的 T-only 对应）**  
  - 模态：`use_audio=true`，`use_text=false`，`use_video=false`，不使用生理信号；  
  - 其余模型结构与训练超参数（注意力模块、损失配置、`batch_size=4`、`num_epochs=50`、`learning_rate=1e-4` 等）与 AT Baseline / T-only 保持一致；  
  - 训练模式同样为三数据集混合预训练，域适应与数据集特定归一化均关闭：`use_domain_adaptation=false`，`dataset_normalization.enabled=false`；  
  - 通过 `tmux` 运行完整 50 个 epoch，避免 SSH 断线导致中途终止。

- **训练收敛情况（TensorBoard `train/loss_classification`）**  
  - 初始阶段（Epoch 0）训练分类损失约为 **2.59 左右**，与 AT/T-only 相近；  
  - 随训练进行，`train/loss_classification` 仅在 **2.47–2.51 区间内小幅波动**，整体下降幅度远小于 AT/T-only，后期甚至略有回升，表明当前配置下音频分支的表示能力有限，学习较为困难。

- **验证集指标整体范围（综合所有 epoch）**  
  - `val/accuracy` 大部分 epoch 维持在 **约 0.08–0.15** 区间，略低于 T-only 的 0.12–0.20 以及 AT Baseline 的 0.19–0.20；  
  - `val/f1` 主要集中在 **约 0.15–0.23 左右**（如 Epoch 1、6、15 等 F1≈0.26，但整体波动较大），最佳值与 T-only 接近但不稳定；  
  - 验证总损失 `val/loss_total` 在 **4.5–8.3** 之间持续偏高，后期长期维持在 8 左右，明显高于 AT/T-only 的 4.x 区间，提示当前 A-only 配置下模型在验证集上存在较明显的欠拟合与数值不稳定。

- **代表性最优 epoch（按验证集 F1 选取）**  
  - 在前 20 个 epoch 内，若干 epoch（如 Epoch 1、6、11、15 等）`val/f1` 接近或略高于 0.25，但对应的 `val/loss_total` 已开始升高，且后续并未进一步改善；  
  - 从整体 50 epoch 视角看，A-only 在验证集上没有出现类似 AT Baseline 那样“稳定收敛到一个较低损失 + 稳定提升的 F1 区间”的现象，而是呈现出“训练损失基本不降、验证损失持续升高、指标在较低水平附近震荡”的模式。

- **与 AT / T-only 基线的对比与分析要点**  
  - **总体性能**：在当前完全对齐的训练配置下，A-only 的 `val/accuracy` 和 `val/f1` 整体均低于 AT Baseline，且与 T-only 相比也没有形成明显优势；  
  - **收敛行为**：AT 与 T-only 都表现出“训练损失稳步下降、验证损失在 4.x 区间波动”的相对健康状态，而 A-only 的训练损失几乎不降、验证损失长期处于 7–8 以上，说明音频特征在当前预处理和模型配置下较难单独支撑有效的跨数据集情绪判别；  
  - **实验结论（供论文模态消融小节使用）**：  
    - AT（音频+文本） > T-only（文本） ≈ A-only（音频）  
    - 文本模态在三数据集混合场景中提供了更稳定、更易优化的判别信息；  
    - 在不引入额外正则、数据增强或专门的音频预处理优化之前，**单独音频模态的预训练效果有限，主要价值在于与文本模态结合后的补充作用**。

#### 10.5.3 AT + 域适应（DA）预训练结果（2026‑03‑12）

本小节记录在 AT 基线配置上**开启域适应模块（DA）**后的三数据集混合预训练结果，对应运行目录：  
`logs/AT_pretrain_3datasets_DA_20260312_201945`。

- **实验配置概要（与 10.5 的 AT Baseline / 10.5.1 / 10.5.2 对应）**  
  - 模态：`use_audio=true`，`use_text=true`，`use_video=false`，不使用生理信号，与 AT Baseline 完全一致；  
  - 域适应相关开关：  
    - `model.domain_adaptation.enabled=true`；  
    - `training.loss.use_domain_adaptation=true`；  
    - `training.loss.domain_loss_weight=0.1`；  
  - 其余模型结构与训练超参数（`batch_size=4`、`num_epochs=50`、`learning_rate=1e-4` 等）保持与 AT Baseline 一致，用于构成“有 / 无 DA”严格可比的一对对照实验；  
  - 训练同样通过 `tmux` 运行完整 50 个 epoch，避免 SSH 断线导致中途终止。

- **训练收敛情况（TensorBoard `train/loss_classification` & `train/loss_total`）**  
  - 初始阶段（Epoch 0）训练总损失约为 **2.49 左右**（`cls_loss≈2.48 + domain_loss≈0.03`），与 AT Baseline 非常接近；  
  - 随着 epoch 增加，训练分类损失从约 **2.48** 稳步下降至 **约 2.13 左右**，整体曲线略高于 T-only，但明显低于 A-only，且与 AT_noDA 相比略有改善；  
  - DA 分支的 `domain_loss` 维持在 **0.01–0.03** 的较低水平，对总损失有一定贡献但未导致明显不稳定。

- **验证集指标整体范围（综合所有 epoch）**  
  - `val/accuracy` 大致分布在 **约 0.07–0.17** 区间，中后期多次达到 **0.14–0.16 左右**；  
  - `val/f1` 主要位于 **约 0.14–0.26** 之间，早期个别 epoch（如 Epoch 5–7、13–15）在 0.23–0.26 附近，后期趋于 **0.14–0.18** 的稳定区间；  
  - 验证总损失 `val/loss_total` 大致在 **约 4.0–5.1** 之间波动，整体水平与 AT Baseline 接近，略高于 T-only、显著低于 A-only。

- **代表性最优 epoch（按验证集 F1 选取）**  
  - 在前 20 个 epoch 内，若干 epoch（如 Epoch 5–7、13–15）`val/f1` 达到 **约 0.23–0.26** 的相对峰值，同时 `val/accuracy` 在 **0.15–0.16** 附近：  
    - 例如 epoch 5–6 一段区间：Loss ≈ **5.0**，Accuracy ≈ **0.16**，F1 ≈ **0.25**；  
  - 在后 30 个 epoch，`val/f1` 整体回落并稳定在 **约 0.14–0.18** 区间，说明 DA 在当前权重和结构下带来的收益有限，需要结合更精细的超参数与正则策略进一步打磨。

- **与 AT_noDA / T-only / A-only 的对比与分析要点**  
  - **与 AT_noDA 对比（同模态 + 同配置）**：  
    - AT_noDA 的典型指标约为：Accuracy≈0.19–0.20，F1≈0.25 左右；  
    - AT_DA 在部分 epoch 上能达到接近 0.25 的 F1，但整体 `val/accuracy` 与 `val/f1` 未出现稳定、显著的提升，整体性能与 AT_noDA 相近；  
    - 从当前结果看，**在不进一步调参的前提下，简单开启 DA 并未在三数据集混合场景下显著改善分类性能**，但也未引入数值不稳定或训练崩溃问题。  
  - **与 T-only / A-only 对比（模态消融角度）**：  
    - 无论是否启用 DA，AT 模态（音频+文本）的综合表现依旧优于 T-only 和 A-only 两条单模态基线；  
    - 这说明 **模态互补带来的收益高于目前 DA 模块在域对齐层面带来的改进**，后续可在“未来工作”中将 DA 的深入调优作为扩展方向。  
  - **实验结论（供论文“域适应消融”小节使用）**：  
    - 在当前网络结构与损失权重配置下，**AT+DA 与 AT_noDA 在整体 Accuracy / F1 上相近，未观察到稳定、显著的收益**；  
    - 这提示对于本任务，域适应模块可能需要更针对性的设计（如单独对特定模态或特定数据集进行对齐），或在更大规模/更明显域差异的数据配置下才会体现优势。

#### 10.5.4 CREMA-D 单数据集微调 vs 从零训练（2026‑03‑13 / 03‑14）

本小节记录「**预训练 + CREMA 微调**」与「**CREMA 从零训练（scratch）**」两条对照实验的完整结果，并对二者及与前述预训练基线进行消融对比。对应运行目录分别为：  
`logs/AT_crema_from_pretrain_noDA_20260313_212316`（预训练+微调）、  
`logs/AT_crema_scratch_noDA_20260314_102323`（scratch 组）。

- **实验配置概要（与 10.5 / 10.5.3 对应）**  
  - 模态：`use_audio=true`，`use_text=true`，`use_video=false`，与 AT Baseline 一致；  
  - 域适应：`use_domain_adaptation=false`，与 10.5 的 noDA 设定一致；  
  - 训练模式：`mode=finetune`，目标数据集仅 CREMA-D（`training.finetune.datasets: [crema]`）；  
  - **预训练+微调 run（AT_crema_from_pretrain_noDA_20260313_212316）**：  
    - 初始化：从 `checkpoints/checkpoint_pretrain_best.pth`（AT Baseline 三数据集预训练最优权重）加载，对分类头等做 `strict=False` 兼容；  
    - 训练轮数：30 epoch，其余超参与 AT Baseline 一致（如 `batch_size=4`、`learning_rate=1e-4` 等）。  
  - **Scratch run（AT_crema_scratch_noDA_20260314_102323）**：  
    - 初始化：随机初始化（或等价地使用不加载预训练权重的 finetune 配置），同一 CREMA-D 数据与 30 epoch 设定；  
    - 用于对比「仅用 CREMA 从零训练」与「三数据集预训练后再在 CREMA 上微调」的收益。

- **训练收敛情况（TensorBoard `train/loss_classification` & metrics.csv）**  
  - **预训练+微调（AT_crema_from_pretrain_noDA_20260313_212316）**：  
    - 根据 TensorBoard 截图，初始训练分类损失约 **2.55**（Epoch 0 附近），从预训练权重起步后在前段 epoch 有较明显下降；  
    - 有记录的 metrics 自 Epoch 10 起（train loss 约 **2.29**），至 Epoch 29 收敛到 **2.240683**（cls_loss），曲线平滑；  
    - 训练时长约 **1.687 小时**（可能与当时机器负载或数据加载有关）。  
  - **Scratch（AT_crema_scratch_noDA_20260314_102323）**：  
    - 初始（Epoch 0）训练分类损失约 **2.28**，最终（Epoch 29）**2.229980**（cls_loss），约 5 个 epoch 内即降至约 2.24 并趋于稳定；  
    - 训练时长约 **31.72 分钟**，墙钟时间短于预训练+微调 run，反映从零训练时在 CREMA 单数据集上迭代效率较高。

- **验证集指标整体范围（综合有记录的 epoch）**  
  - **预训练+微调（AT_crema_from_pretrain_noDA_20260313_212316）**：  
    - `val/accuracy` 稳定在 **0.17–0.18** 左右（如 0.176075、0.178763）；  
    - `val/f1` 在 **0.29–0.30** 区间（如 0.297365、0.300903、0.303307），**明显优于 scratch 组**；  
    - `val/loss_total` 约 **2.97–4.04**，部分 epoch 验证损失较低。  
  - **Scratch（AT_crema_scratch_noDA_20260314_102323）**：  
    - `val/accuracy` 大致在 **0.16–0.19** 区间，Epoch 19–22 可达 **0.178763**，Epoch 23 出现 **0.185484**；  
    - `val/f1` 前期（Epoch 11、19–22）可达 **0.27** 左右，后期（Epoch 23–29）在 **0.21–0.23** 区间波动；  
    - `val/loss_total` 约 **3.5–4.2**，中后期略升。  

  说明：**预训练+微调在 CREMA 验证集上取得了更高的 F1（约 0.30 vs 0.27）**，体现三数据集预训练后再在 CREMA 上微调对目标域泛化性能的增益。

- **代表性最优 epoch（供论文表格使用）**  
  - **预训练+微调（AT_crema_from_pretrain_noDA_20260313_212316）**：Epoch 10/12/16/23/26 等（Accuracy ≈ **0.178763**，F1 ≈ **0.3009–0.3033**）。  
  - **Scratch（AT_crema_scratch_noDA_20260314_102323）**：Epoch 19–20（Accuracy ≈ **0.178763**，F1 ≈ **0.266**）；Epoch 11（F1 ≈ **0.270**）。  
  - 详细数值见 `logs/AT_crema_from_pretrain_noDA_20260313_212316/metrics.csv` 与 `logs/AT_crema_scratch_noDA_20260314_102323/metrics.csv`。

- **与 10.5 / 10.5.1 / 10.5.2 / 10.5.3 的消融对比要点**  
  - **与 AT Baseline 三数据集预训练（10.5）**：  
    - AT Baseline 在混合验证集上约 **Accuracy 0.19–0.20、F1 ≈ 0.25**；CREMA 微调/scratch 均在**单数据集 CREMA-D 验证集**上评估，任务与数据分布不同，不宜直接比较绝对值；  
    - 可比的是：以 AT Baseline 为初始化做 CREMA 微调后，在 CREMA 验证集上 **val/f1 提升至约 0.30**，优于同配置下从零训练（约 0.27），说明**预训练为单数据集微调提供了更好的泛化起点**。  
  - **与 T-only / A-only 预训练（10.5.1 / 10.5.2）**：  
    - T-only 预训练最终 `train/loss_classification` 约 **2.11**，A-only 约 **2.51**；CREMA 微调/scratch 的最终训练损失约 **2.23–2.24**，介于二者之间；  
    - 在 CREMA 单数据集上，**预训练+微调**的验证 F1（**0.30**）优于 scratch（0.27），与「预训练带来更好表示」的结论一致。  
  - **与 AT+DA 预训练（10.5.3）**：  
    - AT+DA 在三数据集上 `train/loss_classification` 可降至约 **2.13**；CREMA 微调/scratch 未启用 DA，在 CREMA 上约 2.23–2.24；  
    - 若后续要做「AT+DA 预训练 → CREMA 微调」，可沿用本节配置，仅将 resume 改为 AT_DA 的 checkpoint，做进一步对比。

- **实验结论（供论文“预训练 vs 从零训练”小节使用）**  
  1. **验证性能**：**预训练+CREMA 微调**在 CREMA-D 验证集上 **val/f1 达约 0.29–0.30**，**优于 CREMA 从零训练（scratch）的约 0.21–0.27**，说明**三数据集预训练后再在 CREMA 上微调能带来明确的泛化收益**。  
  2. **训练时间**：本轮 scratch run 墙钟时间（约 31 分钟）短于预训练+微调 run（约 1.7 小时），可能受机器负载、数据加载或日志写入等环境因素影响；二者均为 30 epoch，可比性主要体现在验证集指标上。  
  3. **工程建议**：若以**验证集 F1 / 准确率为首要目标**，应优先采用「预训练 + CREMA 微调」；若需在资源受限下快速试跑，可接受略低的 F1 时再考虑 scratch 方案。

- **论文写作时如何使用这些数据（建议）**  
  1. **曲线图**：  
     - 绘制两条 run 的 `train/loss_classification` 随 step/epoch 变化（可引用 TensorBoard 截图或从 `metrics.csv` 提取）；  
     - 可选：绘制两条 run 的 `val/f1` 随 epoch 变化，突出预训练+微调组在验证集上的优势。  
  2. **结果表**：  
     - 在「预训练 vs 从零训练」表中列出：预训练+微调（AT_crema_from_pretrain_noDA_20260313_212316）与 Scratch（AT_crema_scratch_noDA_20260314_102323）的 **Epoch 数、总训练时间、最终 train loss、代表性 val Accuracy/F1**；  
     - 可与 10.5 的 AT Baseline、10.5.3 的 AT+DA 并列一列「任务/数据」列（三数据集 vs CREMA-D），避免跨任务直接比绝对值。  
  3. **文字分析要点**：  
     - 强调「预训练+单数据集微调」在 **CREMA-D 验证集 F1** 上相对「从零训练」的**稳定提升（约 0.30 vs 0.27）**；  
     - 可简要说明训练时长差异可能受环境因素影响，以验证指标为主要对比依据。

#### 10.5.5 视频单模态基线（V-only，无 DA）预训练结果（2026‑03‑17）

本小节记录 2026‑03‑17 完成的 **V-only（三数据集混合、无域适应）预训练**结果，对应运行目录：  
`logs/V_only_pretrain_3datasets_noDA_20260317_205923`。

- **实验配置概要（与 10.5 / 10.5.1 / 10.5.2 对应）**  
  - 模态：`use_video=true`，`use_audio=false`，`use_text=false`，`use_physiological=false`；  
  - 融合：`fusion_strategy="standard"`（单模态下融合模块不会引入跨模态交互，但保持配置一致便于复用训练脚本）；  
  - 域适应 / 归一化：`use_domain_adaptation=false`，`dataset_normalization.enabled=false`；  
  - 损失：仅分类任务（`loss_weights.classification=1.0`，回归与趋势预测权重为 0），并启用 `ClassBalancedLoss`；  
  - 训练参数：`batch_size=2`，`learning_rate=1e-4`，`num_epochs=50`；  
  - 视频输入：为避免显存/数值问题，采用轻量设置 `frame_size=112`、`num_frames=4`（详见 `config/config_video_only.yaml`）。

- **训练收敛情况（TensorBoard `train/loss_classification` & `train/loss_total`）**  
  - 训练分类损失在 Epoch 0 为 **2.6578**，到 Epoch 49 为 **2.5621**（`metrics.csv`）；  
  - TensorBoard 中该 run（青色曲线）在 Step 49 的平滑值约 **2.5607**，总用时约 **13.06 小时**，整体呈现“下降幅度有限、后期趋于平台”的收敛形态；  
  - 这表明在当前输入规模与优化设置下，**仅视频模态的可学习信号较弱，训练收敛较慢且幅度不大**。

- **验证集指标整体范围（综合所有 epoch）**  
  - `val/accuracy` 介于 **0.0167–0.0907**；  
  - `val/f1` 介于 **0.0299–0.1396**；  
  - `val/loss_total` 介于 **3.2048–4.8142**。  
  - 指标整体偏低且波动明显，提示 V-only 在三数据集混合预训练上存在较明显的欠拟合/泛化不足。

- **代表性最优 epoch（按验证集指标选取）**  
  - **按 F1 最优（Epoch 0）**：  
    - `val/loss_total` ≈ **3.4051**，Accuracy ≈ **0.0751**，Precision ≈ **1.0000**，Recall ≈ **0.0751**，F1 ≈ **0.1396**；  
  - **按 Accuracy 最优（Epoch 13）**：  
    - `val/loss_total` ≈ **3.9609**，Accuracy ≈ **0.0907**，Precision ≈ **0.3456**，Recall ≈ **0.0907**，F1 ≈ **0.1355**。  
  - 说明：出现“Precision 很高但 Recall 很低”的现象，通常意味着模型倾向预测少数类别或单一类别，在极不平衡的混合数据上难以稳定泛化。

- **与 AT / T-only / A-only / AT+DA 的对比与分析要点（模态消融角度）**  
  - 与 AT Baseline（10.5）相比：V-only 的 `val/accuracy` / `val/f1` 明显更低（AT 通常可到 Accuracy≈0.19–0.20、F1≈0.25），说明在当前三数据集混合设置中，**仅视频信息不足以支撑有效的情绪判别**；  
  - 与 T-only（10.5.1）相比：T-only 的训练损失与验证指标整体显著优于 V-only，提示 **文本模态在跨数据集混合预训练中更稳定**；  
  - 与 A-only（10.5.2）相比：两者都存在优化困难，但 V-only 的验证 Accuracy/F1 整体更低，且收敛幅度有限；  
  - 与 AT+DA（10.5.3）相比：DA 在 AT 上未带来显著提升；在 V-only 场景中，优先级更高的是先把 VT / AVT 跑通并建立“有/无视频”的可比对照，再决定是否值得对视频场景额外启用 DA。

- **实验结论（供论文“视频模态引入动机”小节使用）**  
  1. 在当前输入规模（112×112、4 帧）与训练策略下，V-only 在三数据集混合预训练上表现为“训练损失下降有限、验证指标偏低且波动大”，难以作为强基线；  
  2. 该结果为后续实验提供了清晰动机：**视频模态更可能需要与文本/音频进行互补（VT、AVT）**，或需要更合适的视频特征（更高分辨率/更多帧/更强 backbone/更针对的视频增强）才能发挥作用；  
  3. **（已执行）** 按 8.6.1 已完成 **V-only → VT noDA → AVT noDA**（AVT 主结果见 **10.5.7**）；**下一步**按 8.6.2 推进 **VT+DA / AVT+DA** 的域适应消融，以形成「视频 + DA」对照。

#### 10.5.6 周工作日志与阶段性效果评级（2026‑03‑15 至 2026‑03‑21，含 VT 无DA 完整收敛）

本小节汇总本周（3/15–3/21）围绕“引入视频模态”所完成的实验工作，重点包括 **V-only 无DA完整训练** 与 **VT 无DA完整训练（分段续训后完成 1–50 epoch）**，并给出效果评级与消融对比分析，便于后续周报与论文实验章节直接引用。

- **本周核心工作与过程日志（已完成）**  
  1. 完成 `config/config_video_only.yaml` 的稳定化配置与全量训练（50 epoch）：  
     - 关键调整：`batch_size=2`、`frame_size=112`、`num_frames=4`；  
     - 完整 run：`logs/V_only_pretrain_3datasets_noDA_20260317_205923`。  
  2. 完成 `config/config_VT_noDA.yaml` 的创建、续训修复与全程跑通：  
     - 首段 run（1–16 epoch）：`logs/VT_pretrain_3datasets_noDA_20260318_203248`；  
     - 续训 run（16–50 epoch）：`logs/VT_pretrain_3datasets_noDA_20260321_142141`；  
     - 两段合并后覆盖完整训练周期（1–50 epoch）。  
  3. 训练异常排查与工程修复闭环：  
     - 定位 `moov atom not found` / `Failed to open video file`（如 `meld_train_1166.mp4`）导致训练明显变慢；  
     - 执行 `scripts/check_media_health_dir.py` 完成坏样本定位，为后续清洗与复跑提供依据；  
     - 修复 `scripts/train.py` 在 `--resume` 时未恢复 `scheduler_state_dict` 的问题，避免恢复后学习率轨迹重置导致 TensorBoard 曲线“断崖”。

- **V-only 无DA（对照基线）结果摘要**  
  - run：`V_only_pretrain_3datasets_noDA_20260317_205923`；  
  - 训练终点（Epoch 49）：`train/loss_classification` ≈ **2.5621**（TensorBoard smoothed ≈ **2.5607**），总时长约 **13.06 hr**；  
  - 验证范围（50 epoch）：`val/accuracy` ≈ **0.0167–0.0907**，`val/f1` ≈ **0.0299–0.1396**，`val/loss_total` ≈ **3.2048–4.8142**。

- **VT 无DA（完整 1–50 epoch）结果摘要**  
  - run 组成：`VT_pretrain_3datasets_noDA_20260318_203248`（1–16 epoch）+ `VT_pretrain_3datasets_noDA_20260321_142141`（16–50 epoch）；  
  - 训练损失（train/loss_classification）总体区间约 **2.49–2.72**，后半程（约 Epoch 33 后）有抬升，终点（Epoch 49）≈ **2.6458**；  
  - 验证指标范围（全程）：  
    - `val/accuracy` ≈ **0.024–0.1166**；  
    - `val/f1` ≈ **0.0429–0.1889**；  
    - `val/loss_total` ≈ **2.4758–3.5247**。  
  - **最佳点（续训段 Epoch 34）**：Accuracy ≈ **0.1166**，F1 ≈ **0.1889**；  
  - **终点（Epoch 49）**：Accuracy ≈ **0.0821**，F1 ≈ **0.1218**，`val/loss_total` ≈ **3.4370**。

- **消融对比分析（V-only vs VT，及与历史基线关系）**  
  1. **相对 V-only 的收益（视频+文本互补得到验证）**：  
     - VT 的最优 Accuracy/F1（**0.1166 / 0.1889**）高于 V-only 最优区间上限（约 **0.0907 / 0.1396**）；  
     - 说明引入文本后，较纯视频更容易学到可分辨情绪信号，证明“视频模态需与其它模态互补”的方向是有效的。  
  2. **稳定性与泛化不足仍然存在**：  
     - VT 虽有阶段高点，但终点回落（Epoch 49：Accuracy≈0.0821，F1≈0.1218），未形成持续上升或稳定高平台；  
     - 曲线表现与数据质量（坏视频）和长时训练稳定性高度相关，当前更接近“可用但不稳”的中间阶段。  
  3. **相对 T-only / AT 基线仍偏弱（按既有章节记录）**：  
     - 与 10.5.1 / 10.5（T-only / AT）相比，VT 当前完整结果尚未给出“稳定且显著优于文本或 AT”的证据；  
     - 后续需要在统一清洗后的数据上做复跑，才能公平判断“视频是否带来稳定增益”。

- **实验效果评级（更新为完整结果后）**  
  - **V-only 无DA：C（偏弱）**  
    - 理由：可跑通但指标低，单视频模态难以形成竞争力。  
  - **VT 无DA（1–50 epoch 完成）：C+（有提升但不稳定）**  
    - 理由：相对 V-only 有明显峰值提升，但终点回落、波动偏大，尚未达到稳定强基线水平。  
  - **本周工程执行评级：A-**  
    - 理由：完成从异常定位、续训逻辑修复到 50 epoch 跑通的闭环，实验可复现性与可解释性显著提升。

- **结论与下一步建议**  
  1. **结论**：VT 相较 V-only 已体现“多模态互补收益”，但当前收益不稳定，不能直接认定其已优于 T-only / AT 主基线。  
  2. **工程侧优先级**：在 `train.py` 中保留并严格执行 resume 的 model/optimizer/scheduler 同步恢复，避免再次出现续训断崖干扰结论。  
  3. **实验侧优先级（已更新）**：**AVT noDA 全量 50 epoch 已跑通并记入 10.5.7**；后续优先在统一数据清洗版本上决定是否复跑 VT/AVT 以提升可比性，再进入 **`VT+DA` / `AVT+DA`** 与下游微调，形成“视频 + 域适应”的完整证据链。

#### 10.5.7 AVT 三模态无域适应（noDA）预训练完整结果（2026‑03‑23）

本小节记录 **Audio + Video + Text 全模态、三数据集混合、无 DA** 的一次完整预训练，作为与 AT、VT、V-only 并列的**主结果行**，便于论文「全模态基线」与模态消融对照。

- **运行目录**：`logs/AVT_pretrain_3datasets_noDA_20260323_202809`  
- **配置文件**：`config/config_AVT_noDA.yaml`  
- **复现命令**：`python scripts/train.py --config config/config_AVT_noDA.yaml --mode pretrain`  
- **模态与任务**：`use_audio=true`，`use_video=true`，`use_text=true`，`use_physiological=false`；融合 `fusion_strategy="standard"`；`use_domain_adaptation=false`；损失仅分类（`ClassBalancedLoss`，回归/趋势权重为 0）。  
- **训练超参（与文档 8.6.1 一致）**：`num_epochs=50`，`batch_size=1`（AVT 显存压力最大），`learning_rate=1e-4`；视频 `frame_size=112`，`num_frames=4`。

**训练收敛（`metrics.csv`，`phase=train`）**  
- 分类损失：Epoch 0 约 **2.526**，Epoch 49 约 **2.254**，整体呈缓慢下降。  
- 与 AT Baseline（约 2.25 量级终点）相比，三模态端到端训练更重、优化难度更大，但终点训练损失仍在同一数量级。

**验证集表现（`phase=val`，混合验证集）**  
- **全程范围**：`val/accuracy` 约 **0.075–0.166**；`val/f1` 约 **0.100–0.274**；`val/loss_total` 约 **2.72–6.30**（前半程验证损失偏高、波动大）。  
- **终点（Epoch 49，推荐作为主报告行）**：`val/accuracy` ≈ **0.1663**，`val/f1` ≈ **0.2400**，`val/loss_total` ≈ **3.014**；`precision` ≈ **0.470**，`recall` ≈ **0.166**（形态正常，非单类塌缩）。  
- **关于 CSV 中 Epoch 0–32 附近的 `precision≈1.0` 与较高 F1**：与当时验证集上 **分类指标计算顺序 / 实现** 及类别极不平衡下的数值形态有关；若以论文级 PR/F1 为准，建议对同一 checkpoint 使用 **当前已修复的** `scripts/recompute_val_metrics.py` 复核，或仅将 **Epoch 33 以后** 及 **终点 Epoch 49** 作为主表数字。全表原始峰值 F1 约 **0.273**（多出现在 Epoch 3–32），**不宜单独作为最终结论**，除非完成重算并与 TensorBoard 对齐。

**与既有基线的消融对比（三数据集混合预训练、可比口径）**  

| 设置 | 代表 run / 章节 | 验证集量级（Acc / F1，混合 val） | 简要结论 |
|------|-----------------|-----------------------------------|----------|
| **AT** | 10.5 | Acc ≈ **0.19–0.20**，F1 ≈ **0.25** | 无视频时主基线，指标最高之一 |
| **AVT（本小节）** | 10.5.7 | 终点 Acc ≈ **0.166**，F1 ≈ **0.240** | 全模态可跑通；终点仍 **略低于 AT**，未观察到「简单叠模态必涨」 |
| **VT** | 10.5.6 | 最优 Acc/F1 约 **0.117 / 0.189**，终点更低 | 视频+文本有峰值但不稳 |
| **V-only** | 10.5.5 | Acc/F1 明显更低 | 单视频信号弱 |
| **T-only / A-only** | 10.5.1 / 10.5.2 | 见各节 | 单模态对照 |

**效果评价与结论**  
1. **工程**：在 `batch_size=1`、轻量视频设置下，**AVT 全量 50 epoch 稳定跑完**，日志与 `metrics.csv` 完整，可作为论文「全模态无 DA」一行。  
2. **效果**：终点验证 **F1 仍低于 AT**，说明在当前数据与结构下，**增加视频分支带来优化负担与噪声**，未超过「AT 已较强的文本+音频」组合；与 VT 相比，**AVT 终点 F1（≈0.24）高于 VT 最优峰值（≈0.19）与终点**，表明 **补回音频对三模态情绪判别仍有实质帮助**。  
3. **论文表述建议**：可同时报告「**AT > AVT（终点）> VT 峰值**」这一层次，并说明视频分辨率/帧数/batch 等限制，避免读者误解为「视频无用」，而应表述为「在当前轻量视频设定下，全模态需配合更强视觉表征或清洗后复验」。  

**评级**：**B（可用、有对比价值）**——略低于 AT 的绝对验证指标，但作为全模态主行完整、与 VT/V-only 形成链条，**论文消融价值高**。

**下一步（实验）**：在 8.6.2 中推进 **VT+DA / AVT+DA**，或先对坏样本清洗后复跑 AVT 再对比；下游可做 **AVT → CREMA/MELD 微调**（与 10.5.4 AT 微调口径对齐）。

#### 10.5.8 AVT 三模态 + 域适应（AVT+DA）预训练完整结果（2026‑03‑25 至 2026‑03‑28）

本小节记录 **Audio + Video + Text 全模态、三数据集混合、开启 DA** 的一次完整预训练结果，对应运行目录：  
`logs/AVT_pretrain_3datasets_DA_20260325_215401`。  
该实验用于回答“在与 AVT noDA 基本一致的主干配置下，**引入域对抗训练是否带来稳定收益**”。

- **实验配置概要（与 10.5.7 保持可比）**  
  - 模态：`use_audio=true`，`use_video=true`，`use_text=true`，`use_physiological=false`；  
  - 融合：`fusion_strategy="standard"`；  
  - 域适应：`model.domain_adaptation.enabled=true`，`training.loss.use_domain_adaptation=true`，`domain_loss_weight=0.1`；  
  - 数据集归一化：`dataset_normalization.enabled=false`（保持与 noDA 主线一致，避免额外变量干扰）；  
  - 损失：分类主任务 + 域损失（回归/趋势权重为 0）；  
  - 训练参数：`num_epochs=50`，`batch_size=1`，`learning_rate=1e-4`；视频输入仍为 `frame_size=112`、`num_frames=4`（见 `config/config_AVT_DA.yaml`）。

- **训练过程日志（关键阶段）**  
  1. **冷启动阶段（Epoch 0–8）**：分类损失约 `2.53–2.56`，验证 Accuracy/F1 处于低位（`Acc≈0.04–0.12`，`F1≈0.009–0.045`）。  
  2. **域对抗拉锯阶段（Epoch 9–12）**：`domain_loss` 出现明显峰值（约 `0.56–0.69`），对应训练“先对抗、再平衡”的常见形态。  
  3. **中后期稳定阶段（Epoch 26 以后）**：`domain_loss` 回落并稳定在约 `0.18–0.22`，验证 Accuracy 逐步抬升到 `0.15+` 区间。  
  4. **收官阶段（Epoch 42–49）**：验证指标保持在相对稳定区间，未出现明显崩溃或发散，50 epoch 完整跑通。

- **训练收敛（`metrics.csv`，`phase=train`）**  
  - 总损失：Epoch 0 为 **2.5670**，Epoch 49 为 **2.4716**；  
  - 分类损失：Epoch 0 为 **2.5605**，Epoch 49 为 **2.4507**，整体缓慢下降；  
  - 域损失：从中期峰值（约 **0.6936**）回落到 Epoch 49 的 **0.2085**，说明域分支从强对抗逐步进入稳定平衡。

- **验证集表现（`phase=val`，完整 50 epoch）**  
  - **全程范围**：  
    - `val/accuracy` 约 **0.0367–0.1755**；  
    - `val/f1` 约 **0.0094–0.1525**；  
    - `val/loss_total` 约 **2.1630–4.3561**。  
  - **终点（Epoch 49）**：  
    - `val/loss_total` ≈ **3.2787**；  
    - `val/accuracy` ≈ **0.1641**；  
    - `val/precision` ≈ **0.1922**；  
    - `val/recall` ≈ **0.1641**；  
    - `val/f1` ≈ **0.1403**。  
  - **代表性最佳点**：  
    - **Accuracy 最优（Epoch 44）**：Acc ≈ **0.1755**，F1 ≈ **0.1525**；  
    - **F1 最优（Epoch 44）**：F1 ≈ **0.1525**，Acc ≈ **0.1755**。

- **与 AVT noDA / AT / VT 的消融对比（当前口径）**  
  1. **AVT+DA vs AVT noDA（10.5.7）**  
     - 终点对比：AVT+DA 的 Acc（≈0.164）与 AVT noDA（≈0.166）接近，但 F1（≈0.140）明显低于 AVT noDA（≈0.240）；  
     - 说明：在当前轻量视频设置与混合数据场景下，**引入 DA 并未带来稳定增益，且对 F1 有明显压制**。  
  2. **AVT+DA vs VT noDA（10.5.6）**  
     - AVT+DA 终点 F1（≈0.140）仍高于 VT noDA 终点（≈0.122），说明加入音频后三模态仍有互补价值；  
     - 但 AVT+DA 的提升幅度不足以超过 AVT noDA，提示主要瓶颈可能来自域对抗与任务主目标的权衡。  
  3. **AVT+DA vs AT 基线（10.5）**  
     - 与 AT 常见较优区间（Acc≈0.19–0.20、F1≈0.25）相比，AVT+DA 仍偏弱，反映“视频 + DA”组合在当前条件下尚未转化为更强判别收益。

- **效果分析与原因判断（实验层）**  
  1. **DA 分支已成功接入且训练可持续**：从曲线与 loss 分解看，域分支正常参与优化，不是“模块失效”问题；  
  2. **主任务收益不足**：当前 `domain_loss_weight=0.1` 可能使特征更偏向域不变性，牺牲了部分类别判别边界；  
  3. **视频分支质量与规模约束仍在**：`112x112/4帧/batch=1` 是工程可运行折中，但可能限制了 AVT+DA 的上限；  
  4. **指标口径注意**：与早期历史 run 一样，论文最终表格建议统一通过 `recompute_val_metrics.py` 对关键 checkpoint 复核，避免不同阶段日志口径差异干扰结论。

- **实验结论（可直接用于论文“DA 消融”小节）**  
  1. 在当前 AVT 训练设置下，**AVT+DA 已完整跑通且稳定收敛**；  
  2. 就本次 50 epoch 结果看，AVT+DA **未优于 AVT noDA**，尤其在 F1 上明显下降；  
  3. 该结果支持一个重要结论：**DA 在三模态混合预训练中并非必然收益，需要结合损失权重、特征质量与数据清洗程度做精细化调参**。

- **下一步建议（与 8.6.2 对齐）**  
  1. 保持 AVT noDA 作为主基线，新增 **AVT+DA 调参小网格**（优先尝试 `domain_loss_weight`：0.02 / 0.05 / 0.1）；  
  2. 在统一口径下做 noDA/DA 终点与 best checkpoint 重算对比，形成论文 DA 消融表；  
  3. 按计划推进 **AVT_noDA_emotion_shift**，补全“融合策略 vs 域适应”两个正交维度的实验矩阵；  
  4. 资源允许时对视频分支做轻度增强（帧数/分辨率/坏样本清洗后复跑），验证 AVT+DA 是否受限于视觉表征质量。

### 10.6 AT Baseline 之后的下一步实验操作指引（2026‑03‑07 更新）

在 AT 基线已经完整预训练并得到最优 checkpoint 的基础上，后续实验建议按“**先模态消融，再逐步加复杂模块**”的顺序推进，便于论文中逐层对比。

> **与当前进度的关系（2026‑03 更新）**：T-only / A-only、V-only、VT noDA、**AVT noDA（10.5.7）**、**AVT+DA（10.5.8）** 均已跑通并记入第十章；本节步骤 1–3 仍作为 **AT 起点上的复现与扩展清单**；含视频下一步建议聚焦 **10.5.8 的 DA 调参复验** 与 **AVT_noDA_emotion_shift**。

- **步骤 1：基线 AT 结果固化与复现检查**  
  - 确认本次 run 的关键信息已经记录：  
    - 运行目录：`logs/multimodal_emotion_recognition_20260305_234413`；  
    - 最优模型：`checkpoints/checkpoint_pretrain_best.pth`；  
    - 配置文件：当时使用的 `config/config.yaml` 已纳入版本管理。  
  - 如需在新环境复现 AT Baseline，只需：  
    1. 同步代码与配置；  
    2. 按 `EXPERIMENT_ENV_SETUP.md` 创建 Conda 环境与数据目录；  
    3. 运行：`python scripts/train.py --config config/config.yaml --mode pretrain`，并检查 TensorBoard 曲线是否与本节描述的趋势大致一致。

- **步骤 2：单模态消融预训练（T-only 与 A-only）**  
  - 目的：在相同训练策略下，分别只启用文本 / 只启用音频，量化各自贡献，为论文“模态消融”小节提供对比表格。  
  - 建议具体操作：  
    1. 在当前 `config/config.yaml` 基础上，保存两份备份，例如：  
       - 文本基线：`config/config_text_only.yaml`；  
       - 音频基线：`config/config_audio_only.yaml`。  
    2. 在 `config_text_only.yaml` 中：  
       - 将 `model.modalities.use_text` 设为 `true`（保持不变），`model.modalities.use_audio=false`，`use_video=false`，保持其余训练参数与 AT Baseline 一致；  
    3. 在 `config_audio_only.yaml` 中：  
       - 将 `model.modalities.use_audio=true`，`model.modalities.use_text=false`，`use_video=false`，保持其余参数一致；  
    4. 分别运行：  
       - 文本基线：`python scripts/train.py --config config/config_text_only.yaml --mode pretrain`；  
       - 音频基线：`python scripts/train.py --config config/config_audio_only.yaml --mode pretrain`。  
  - 记录要求：  
    - 每次运行结束后，在本章第十节中补充一小段文字，写明：实验时间、模态配置（T / A）、最优 epoch 上的 Accuracy / F1、与 AT Baseline 的差异。  
    - 这些结果将在论文的“模态消融实验”表中与 AT Baseline 一起展示。

- **步骤 3：在 AT Baseline 上进行单数据集微调（可选，后续周推进）**  
  - 目的：验证“三数据集混合预训练 + 单数据集微调”是否优于“单数据集从零训练”，为论文中的“预训练收益分析”提供证据。  
  - 建议思路：  
    - 以 `checkpoints/checkpoint_pretrain_best.pth` 作为初始化权重，在配置中指定目标数据集（例如先选 CREMA-D），运行微调模式（`--mode finetune` 或配置中已有的微调选项）；  
    - 同时对比一组“从随机初始化训练”的结果，使用相同的训练轮数和优化器设置；  
    - 在文档与论文中，以表格形式对比两者在目标数据集上的 Accuracy / F1 提升幅度。  
  - 该部分的具体命令行参数与实现细节，可在正式开始微调前单独再规划。该小节仅作为本周工作结束时的“下一步行动蓝图”。本周 PPT 编写指导（3.9–3.14）与图片占位见 10.7。 


## 十二、毕业论文写作与实验策略总指南（基于当前真实实验进度）

本章用于解决“实验做了很多，但论文难以结构化呈现”的问题。核心目标是：把你已经完成与即将完成的实验，转化为**可复现、可对比、可写作**的一套证据链。

### 12.1 先给结论：你目前的实验主线是什么

1. **主模型唯一且稳定**  
   - 所有已完成实验均基于 `models/multimodal_model.py` 的 `MultimodalEmotionModel`。  
   - 训练脚本统一为 `scripts/train.py`。  
   - 也就是说：你不是在“换不同模型”，而是在**同一模型骨架下做配置驱动消融**。
2. **当前已形成的证据链（已完成）**  
   - 模态主线：`T` / `A` / `V` / `AT` / `VT` / `AVT_noDA` / `AVT_DA`（已完成）；
   - 训练范式：预训练 + 下游微调（CREMA，已完成 from_pretrain vs scratch）；  
   - 域适应：AT 的 noDA vs DA、AVT 的 noDA vs DA（已完成）；  
   - 工程可靠性：断点续训、scheduler 恢复、坏视频排查流程已形成 SOP。
3. **当前待补充的关键实验（在做）**  
   - `AVT_noDA`（已完成，见 10.5.7）；  
   - `AVT_DA`（已完成，见 10.5.8）；  
   - `AVT_noDA_emotion_shift`（已配置）。  

> 论文写作原则：先围绕“同一模型骨架 + 配置开关消融”讲清楚，再讲“是否需要换更复杂模型”。

### 12.2 论文整体结构（可直接套用）

建议论文主体章节如下（你可按学校模板调整编号）：

- **第1章 绪论**  
  - 研究背景与意义（多模态情绪识别、跨数据集泛化难点）；  
  - 研究问题定义（多源异构、类别不平衡、域偏移、模态互补）；  
  - 主要工作与贡献（与你实际实现一一对应）；  
  - 论文结构安排。

- **第2章 相关技术与研究现状**  
  - 多模态情绪识别方法（音频/文本/视频）；  
  - 融合策略（早期融合、后期融合、注意力融合）；  
  - 域适应与类别不平衡处理（GRL、Class Balanced/Focal）；  
  - 现有方法不足与本文切入点。

- **第3章 方法设计（核心）**  
  - 整体框架：`MultimodalEmotionModel`；  
  - 各模态特征提取器：ResNet50 / Wav2Vec2 / BERT；  
  - 融合模块：`standard` / `emotion_shift` / `leader_follower` / `two_stage`（可插拔）；  
  - 训练目标：分类损失主导，回归/趋势可选，DA 可选；  
  - 工程实现要点：配置驱动、日志体系、断点恢复。

- **第4章 实验设置与结果分析（核心）**  
  - 环境与实现细节；  
  - 数据集与统一预处理；  
  - 基线结果；  
  - 模态消融；  
  - DA 与损失消融；  
  - 预训练收益（from_pretrain vs scratch）；  
  - 异常复盘与可靠性分析（scheduler 恢复、坏视频处理）。

- **第5章 总结与展望**  
  - 本文结论；  
  - 局限性；  
  - 后续工作（更强视频骨干、更多数据清洗、跨域泛化增强）。

### 12.3 第4章（实验章）写作模板与落地内容

#### 12.3.1 实验环境与实现细节

- 硬件（GPU/CPU/内存）；  
- 软件版本（Python、PyTorch、Transformers、CUDA）；  
- 代码入口：`scripts/train.py`；  
- 配置入口：`config/*.yaml`；  
- 记录方式：TensorBoard + `metrics.csv` + checkpoint。

#### 12.3.2 数据集与统一预处理

- 数据集：CREMA-D、MELD、CMU-MOSEI；  
- 统一目录：`data/train|val|test/{video,audio,text,labels}`；  
- 标签统一到 7 类情绪空间；  
- 数据健康检查：`scripts/check_media_health_dir.py`；  
- 异常样本处理原则（坏视频归档后复跑）。

#### 12.3.3 模型配置与训练策略（必须强调“同骨架可比性”）

- 固定骨架：`MultimodalEmotionModel`；  
- 变量仅来自配置：  
  - 模态开关：`use_video/use_audio/use_text`；  
  - 融合策略：`fusion_strategy`；  
  - DA 开关：`model.domain_adaptation.enabled` 与 `training.loss.use_domain_adaptation`；  
  - 训练超参数：batch、lr、frame_size、num_frames。  
- 为保证公平比较：每次消融只改一个主变量，其余保持一致。

### 12.4 消融实验总体设计（科学、可执行、可写）

#### 12.4.1 变量分层设计（推荐）

1. **一级变量：模态组合**（最核心）  
   - 单模态：T / A / V  
   - 双模态：AT / VT（VA 可选）  
   - 三模态：AVT  
2. **二级变量：域适应**  
   - noDA vs DA（先在 AT、VT、AVT 上做对照）  
3. **三级变量：融合策略**  
   - `standard` vs `emotion_shift`（再扩展到 leader_follower / two_stage）  
4. **四级变量：训练范式**  
   - from_pretrain vs scratch（优先在单数据集微调上比较）

#### 12.4.2 推荐实验顺序（按证据链推进）

**阶段 P0：已完成结果固化**  
- 固化 AT/T/A/V/VT 与 CREMA 微调结论；  
- 清理日志命名和文档对应关系。

**阶段 P1：完成三模态主线**  
1. `AVT_noDA`（`config/config_AVT_noDA.yaml`）  
2. `AVT_DA`（`config/config_AVT_DA.yaml`）  
3. `AVT_noDA_emotion_shift`（`config/config_AVT_noDA_emotion_shift.yaml`）

**阶段 P2：做最小充分消融表**  
- 模态表：T/A/V/AT/VT/AVT（统一指标：best/last 的 Acc/F1/loss）；  
- DA表：AT_noDA vs AT_DA、VT_noDA vs VT_DA、AVT_noDA vs AVT_DA；  
- 融合表：AVT_standard vs AVT_emotion_shift（资源允许再加两种融合）。

**阶段 P3：做下游迁移证据**  
- 继续扩展 from_pretrain vs scratch（CREMA 已有，可补 MELD/MOSEI）。  

**阶段 P4：增强可信度（可选）**  
- 对关键结论组（如 AVT_noDA vs AVT_DA）重复 2~3 次（不同随机种子），报告均值±方差。

### 12.5 论文实验结果应如何呈现（直接照抄这个结构）

#### 12.5.1 表1：主结果总览（建议）

列建议：  
`实验名 | 模态 | 融合 | DA | 预训练/微调 | Best Acc | Best F1 | Last Acc | Last F1 | 训练时长 | 结论`

#### 12.5.2 表2：模态消融（核心）

`T / A / V / AT / VT / AVT` 在同一评价标准下对比，正文回答三个问题：

1. 文本是否仍是最稳模态？  
2. 视频单模态为何弱、与文本结合后是否改善？  
3. AVT 是否稳定优于 AT/VT，还是提升有限且波动更大？

#### 12.5.3 表3：DA 消融

固定模态，仅比较 noDA 与 DA。正文回答两个问题：

1. DA 是否稳定提升 F1，而不是偶然峰值？  
2. DA 是否增加了训练不稳定或收敛成本？

#### 12.5.4 表4：融合策略消融

固定模态（建议 AVT），比较 `standard` 与 `emotion_shift`。正文回答：

1. 高级融合是否带来可重复收益？  
2. 收益是否覆盖其复杂度与训练成本？

#### 12.5.5 图表建议

- 图1：关键 run 的 `train/loss_classification` 曲线；  
- 图2：关键组 `val/f1` 曲线；  
- 图3：恢复训练案例（说明 scheduler 恢复前后差异）；  
- 图4：坏样本处理前后训练耗时对比（若有记录）。

### 12.6 你当前可直接执行的“最小论文闭环实验包”

为保证按期写论文，建议优先完成以下最小包：

1. 模态主线：`T/A/V/AT/VT/AVT_noDA`（完成“模态贡献”论证）；  
2. DA主线：`AT_noDA vs AT_DA`、`VT_noDA vs VT_DA`、`AVT_noDA vs AVT_DA`；  
3. 融合主线：`AVT_noDA_standard vs AVT_noDA_emotion_shift`；  
4. 迁移主线：`from_pretrain vs scratch`（CREMA 已完成，建议补 1 组 MELD）。

只要以上完成，你的实验章就是完整且有说服力的。

### 12.7 写作时的高频问题与统一口径

1. **“我是不是换了很多模型？”**  
   - 统一口径：没有换主模型骨架，始终是 `MultimodalEmotionModel`，变化来自配置开关。
2. **“为什么有的实验效果波动大？”**  
   - 统一口径：多源异构 + 视频质量问题 + 小 batch 训练导致方差偏大；因此加入了数据健康检查与恢复规范。
3. **“为什么有时峰值高但终点回落？”**  
   - 统一口径：存在训练后期过拟合/不稳定，论文应同时报告 best 与 last，避免只报峰值。
4. **“断点续训是否可信？”**  
   - 统一口径：已修复 scheduler 恢复逻辑，恢复后学习率轨迹连续，曲线可解释。

### 12.8 论文摘要、结论可复用框架（提纲）

#### 摘要（建议结构）

- 背景：多模态情绪识别跨数据集泛化难；  
- 方法：提出配置驱动统一框架（同骨架、多消融开关）；  
- 实验：在 CREMA-D/MELD/MOSEI 混合场景完成模态、DA、融合策略、预训练收益实验；  
- 结论：文本与多模态互补有效，视频需与其他模态协同，DA/融合策略收益依赖配置与数据质量；  
- 价值：形成可复现实验流程与工程闭环。

#### 结论（建议结构）

- 结论1：统一模型骨架 + 配置驱动消融可系统评估模态贡献；  
- 结论2：单视频模态弱，视频与文本/音频组合可提升但稳定性仍需优化；  
- 结论3：DA 与高级融合并非必然收益，需在严格对照下验证；  
- 结论4：工程规范（tmux、checkpoint、scheduler 恢复、坏样本治理）直接影响实验可信度。

---

## 十三、执行清单与交付标准（论文前最后核对）

### 13.1 实验执行清单（建议逐项打勾）

- [ ] 模态主线结果已齐全：T/A/V/AT/VT/AVT  
- [ ] DA主线结果已齐全：AT、VT、AVT 的 noDA vs DA  
- [ ] 融合主线至少完成 1 组：standard vs emotion_shift  
- [ ] from_pretrain vs scratch 至少两组数据集对比  
- [ ] 每组实验都有：配置文件、run 名、metrics.csv、TensorBoard 截图、文字结论  
- [ ] 异常实验有复盘记录（原因、修复、影响）

### 13.2 论文交付清单（建议）

- 实验章主表（总表 + 三类消融表）；  
- 关键曲线图（loss/f1）；  
- 配置附录（核心字段）；  
- 复现实验说明（命令 + 配置路径 + checkpoint 说明）；  
- 风险与局限性说明（数据质量、计算资源、方差问题）。

### 13.3 最终建议

你当前已经具备完整论文实验基础。接下来最重要的不是“再开更多新坑”，而是：

1. 按 12.4 的顺序把关键对照跑完整；  
2. 按 12.5 的模板统一整理结果；  
3. 按 12.8 的口径写摘要与结论；  
4. 严格做到“结论只基于已完成且可复现的实验”。

只要按本章执行，你就能把当前分散的实验工作，稳定转化为一套结构清晰、证据完整、可直接写入毕业论文的实验体系。
