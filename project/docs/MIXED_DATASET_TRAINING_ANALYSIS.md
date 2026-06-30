# 混合数据集训练问题分析与优化方案

## 一、问题分析

### 1.1 数据集异构性问题

本项目使用的三个开源数据集（CREMA-D、MELD、CMU-MOSEI）存在显著的异构性，主要体现在以下几个方面：

#### 1.1.1 数据分布差异

**CREMA-D数据集**：
- **情感类别数**：6种（Happy, Sad, Angry, Fear, Disgust, Neutral）
- **数据来源**：91名演员在实验室环境下的表演
- **情感表达特点**：较为夸张、标准化
- **数据规模**：约7,442个样本
- **类别分布**：相对平衡，但Disgust类别样本较少

**MELD数据集**：
- **情感类别数**：7种（Anger, Disgust, Sadness, Joy, Neutral, Surprise, Fear）
- **数据来源**：Friends TV系列的真实对话场景
- **情感表达特点**：自然、真实，多说话者交互
- **数据规模**：约13,000个样本
- **类别分布**：Neutral占主导（约40%），其他类别分布不均

**CMU-MOSEI数据集**：
- **情感类别数**：7种（标准情感类别）
- **数据来源**：YouTube视频，真实世界场景
- **情感表达特点**：多样化，包含各种真实场景
- **数据规模**：约3,228个样本
- **类别分布**：高度不平衡，某些类别样本极少

#### 1.1.2 域偏移（Domain Shift）

**域差异分析**：

1. **CREMA-D域**：
   - 环境：实验室，受控条件
   - 光照：标准化
   - 背景：简单、一致
   - 情感表达：标准化、夸张
   - 特征分布：集中、方差小

2. **MELD域**：
   - 环境：电视节目场景
   - 光照：自然光，变化较大
   - 背景：复杂，多说话者
   - 情感表达：自然、真实
   - 特征分布：中等方差

3. **CMU-MOSEI域**：
   - 环境：真实世界，多样化
   - 光照：自然光，变化极大
   - 背景：复杂，多样化场景
   - 情感表达：多样化
   - 特征分布：高方差，分布广泛

**域偏移影响**：
- 不同域的特征分布差异导致模型在不同数据集上表现不一致
- 模型可能学习到域特定的特征而非通用特征
- 跨域泛化能力下降

#### 1.1.3 标注不一致

**标注差异**：

1. **标注标准差异**：
   - CREMA-D：基于演员表演的标准化标注
   - MELD：基于对话上下文的标注
   - CMU-MOSEI：基于视频内容的标注

2. **标注者主观性**：
   - 不同数据集的标注者可能对同一情感有不同的理解
   - 标注边界模糊（如"中性"vs"平静"）

3. **类别定义差异**：
   - MELD使用"joy"而CREMA-D使用"happy"
   - 虽然语义相似，但标注标签不同

#### 1.1.4 数据质量差异

**视频质量**：
- CREMA-D：高分辨率，标准化格式（.flv）
- MELD：中等分辨率，MP4格式
- CMU-MOSEI：分辨率变化大，MP4格式

**音频质量**：
- CREMA-D：高质量录音，标准化采样率
- MELD：从视频提取，质量中等
- CMU-MOSEI：质量变化大，包含噪声

**文本质量**：
- CREMA-D：占位文本或转录文本
- MELD：真实对话文本，质量高
- CMU-MOSEI：转录文本，质量中等

### 1.2 对模型准确性的潜在影响

#### 1.2.1 负面影响

1. **域混淆（Domain Confusion）**：
   - **问题**：模型可能学习到数据集特定的特征（如CREMA-D的实验室背景、MELD的对话特征）而非通用的情感特征
   - **影响**：在测试集上表现差，泛化能力弱
   - **严重程度**：高

2. **类别不平衡（Class Imbalance）**：
   - **问题**：不同数据集的情感类别分布差异大，某些类别在某些数据集中样本极少
   - **影响**：模型偏向多数类别，少数类别识别准确率低
   - **严重程度**：中-高

3. **标注噪声（Label Noise）**：
   - **问题**：不同数据集的标注不一致，导致模型学习困难
   - **影响**：模型收敛慢，准确率上限降低
   - **严重程度**：中

4. **过拟合风险（Overfitting Risk）**：
   - **问题**：模型可能记忆数据集特定的模式（如特定背景、特定说话者风格）
   - **影响**：在训练集上表现好，但在新数据上表现差
   - **严重程度**：中

#### 1.2.2 正面影响

1. **数据多样性（Data Diversity）**：
   - **优势**：更多样化的数据可能提高模型的泛化能力
   - **效果**：模型能够学习到更鲁棒的特征表示

2. **鲁棒性（Robustness）**：
   - **优势**：在不同域上训练可能提高模型对域变化的鲁棒性
   - **效果**：模型能够适应不同的数据分布

### 1.3 问题严重性评估

| 问题类型 | 严重程度 | 影响范围 | 优先级 |
|---------|---------|---------|--------|
| 域偏移 | 高 | 全局 | 高 |
| 类别不平衡 | 中-高 | 分类任务 | 高 |
| 标注不一致 | 中 | 全局 | 中 |
| 数据质量差异 | 中 | 特征提取 | 中 |
| 过拟合风险 | 中 | 泛化能力 | 中 |

## 二、优化方案详细设计

### 2.1 数据集平衡采样（Dataset Balancing）

**目标**：确保每个数据集在每个batch中都有代表，避免模型偏向大规模数据集

**实现策略**：
1. **平衡采样器（BalancedDatasetSampler）**：
   - 确保每个batch包含来自不同数据集的样本
   - 根据数据集大小调整采样权重
   - 支持按比例采样和均匀采样两种模式

2. **加权采样（Weighted Sampling）**：
   - 为每个数据集分配权重
   - 权重可以根据数据集大小、质量等因素调整
   - 支持动态调整权重

### 2.2 类别平衡损失（Class-Balanced Loss）

**目标**：处理类别不平衡问题，提高少数类别的识别准确率

**实现策略**：
1. **类别平衡损失（ClassBalancedLoss）**：
   - 根据类别频率计算权重
   - 对少数类别给予更高权重
   - 支持每个数据集独立的类别权重

2. **Focal Loss**：
   - 关注难分类样本
   - 自动调整难易样本的权重
   - 提高模型对困难样本的学习能力

3. **数据集加权损失（Dataset-Weighted Loss）**：
   - 为不同数据集分配不同的损失权重
   - 可以根据数据集质量、标注可靠性等因素调整

### 2.3 域适应（Domain Adaptation）

**目标**：学习域不变特征，提高跨域泛化能力

**实现策略**：
1. **域分类器（Domain Classifier）**：
   - 识别样本来自哪个数据集（域）
   - 与主任务分类器对抗训练
   - 迫使特征提取器学习域不变特征

2. **域对抗训练（Domain Adversarial Training）**：
   - 使用梯度反转层（Gradient Reversal Layer）
   - 特征提取器最大化域分类错误
   - 域分类器最小化域分类错误

3. **域特定归一化（Domain-Specific Normalization）**：
   - 为每个数据集维护独立的归一化统计
   - 在特征提取后应用数据集特定的归一化
   - 减少域间的特征分布差异

### 2.4 混合训练策略（Mixed Training Strategy）

**目标**：优化混合数据集训练流程，减少域混淆

**实现策略**：
1. **交替训练（Alternating Training）**：
   - 每个epoch交替使用不同数据集
   - 确保模型充分学习每个数据集的特征
   - 减少域混淆

2. **渐进式训练（Progressive Training）**：
   - 先单数据集训练，再混合训练
   - 逐步增加数据集的多样性
   - 提高模型的学习稳定性

3. **课程学习（Curriculum Learning）**：
   - 从简单数据集到复杂数据集
   - 逐步增加训练难度
   - 提高模型的学习效率

## 三、实施计划

### 3.1 高优先级（必须实现）

1. **数据集平衡采样**：确保训练过程的公平性
2. **类别平衡损失**：处理类别不平衡问题
3. **数据集特定归一化**：减少域间特征分布差异

### 3.2 中优先级（推荐实现）

4. **域适应机制**：提高跨域泛化能力
5. **混合训练策略**：优化训练流程

### 3.3 低优先级（可选）

6. **元学习**：快速适应新数据集（未来扩展）

## 四、预期效果

### 4.1 定量指标

- **跨域准确率提升**：在跨数据集测试中，准确率提升5-10%
- **类别平衡改善**：少数类别的F1分数提升10-15%
- **泛化能力增强**：在新数据集上的表现提升8-12%

### 4.2 定性改善

- 模型学习到更通用的特征表示
- 减少对特定数据集的过拟合
- 提高模型的鲁棒性和可靠性

## 五、使用建议

1. **预训练阶段**：优先启用数据集平衡采样、类别平衡损失和数据集特定归一化；域适应和复杂融合模块可以在基础版本稳定后再逐步加入。
2. **微调阶段**：根据目标数据集特点，选择性使用优化策略（例如在单一数据集上可以关闭域适应，只保留类别平衡损失）。
3. **评估阶段**：进行跨数据集评估，验证泛化能力，并对不同配置下的结果进行系统性对比。

---

## 六、当前实验基线与后续消融计划（实测记录）

> 本节用于**真实记录当前混合数据集训练的基线配置与后续实验计划**，方便你在后续多次实验中保持一致性，也方便论文撰写时回溯配置。

### 6.1 当前运行中的 Baseline 配置（无域适应 + 标准注意力）

截至 2026‑03‑05 晚，运行中的“实验一：三数据集混合预训练”采用如下**稳定基线**：

- **数据集与采样**：
  - 训练数据：`["crema", "meld", "mosei"]` 三数据集混合，统一目录结构为 `data/train|val|test/{video,audio,text,labels,...}`。
  - 启用 `BalancedDatasetSampler`：
    - `training.sampling.enabled: true`
    - `training.sampling.mode: "proportional"`
    - 典型统计（当前实际日志）：`dataset_sizes: {meld≈10000, crema≈6000}`，每个 batch 大致 `meld:crema = 3:1`。

- **模型与模态配置**（`config/config.yaml`）：
  - 模态开关：
    - `model.modalities.use_video: true`
    - `model.modalities.use_audio: true`
    - `model.modalities.use_text: true`
    - `model.modalities.use_physiological: false`
  - 视频分支（为适应显存）：  
    - `data.video.frame_size: 160`（原计划为 224）  
    - `data.video.num_frames: 8`（原计划为 16）  
    - backbone：`resnet50` 预训练权重。
  - 音频分支：`facebook/wav2vec2-base`；
  - 文本分支：`bert-base-uncased`。

- **融合与正则**：
  - 当前 Baseline 使用**标准多头注意力融合**：
    - `model.attention.fusion_strategy: "standard"`
    - `model.attention.num_layers: 3`, `num_heads: 8`, `hidden_dim: 512`。
  - EmotionShift / Leader-Follower / Two-Stage 等高级融合策略暂时关闭，保留为后续消融实验。

- **域适应与归一化**：
  - 为先保证训练流程稳定，**目前关闭域适应模块**：
    - `model.domain_adaptation.enabled: false`
    - `training.loss.use_domain_adaptation: false`
  - **仍然开启数据集特定归一化**，以缓解域偏移：
    - `model.dataset_normalization.enabled: true`

- **损失函数与训练超参**：
  - 分类损失：开启类别平衡损失  
    - `training.loss.use_class_balanced: true`
    - `training.loss.use_focal_loss: false`
  - 训练批大小与学习率：  
    - `training.batch_size: 4`（从 16 → 8 → 4 调整，以解决显存问题）  
    - `training.learning_rate: 1e-4`，`optimizer: "adamw"`，`scheduler: "cosine"`  
    - `training.num_epochs: 50`（实际可根据 loss 曲线收敛情况提前或延后停止）。

**论文描述建议**：  
可以将该版本称为：

> “在不引入域对抗模块和复杂情感转变机制的前提下，采用标准多头注意力融合、类别平衡损失与数据集特定归一化的三数据集合并训练基线模型。”

### 6.2 针对 Baseline 的后续消融实验计划

在上述 Baseline 稳定收敛后，建议按“每次只改少量开关”的原则，逐步增加复杂度做消融对比。

#### 6.2.1 域适应模块的有 / 无 对比

- **实验 A（当前 Baseline）**：
  - `model.domain_adaptation.enabled: false`
  - `training.loss.use_domain_adaptation: false`

- **实验 B（仅开启域适应）**：
  - 在保持其他配置与实验 A 完全一致的前提下，仅修改：
    - `model.domain_adaptation.enabled: true`
    - `training.loss.use_domain_adaptation: true`

对比要点：

- 验证集 `F1`、`Accuracy` 是否有提升；
- loss 曲线是否更平滑、是否引入新的不稳定因素；
- 论文中可在“域适应消融”小节给出 A vs B 的表格与曲线对比。

#### 6.2.2 融合策略：Standard vs EmotionShift

在确认“带域适应版本”本身训练稳定后（实验 B 跑通），可以在同一配置上切换融合策略做对比：

- **实验 C：EmotionShift 融合**：
  - 在实验 A 或 B 的基础上，仅修改：
    - `model.attention.fusion_strategy: "emotion_shift"`
  - 其余超参（batch_size、frame_size、learning_rate 等）保持不变。

对比要点：

- 与 `fusion_strategy: "standard"` 相比，EmotionShift 对 `val/f1`、`val/accuracy` 的影响；
- 是否需要适当减小 `model.attention.num_layers` 或 `hidden_dim` 来保证显存与稳定性。

#### 6.2.3 损失设计：ClassBalanced vs Focal / 多数据集加权

在结构性模块稳定后，可以进一步在损失函数维度做消融：

- **实验 D：标准交叉熵（去掉类别平衡）**：
  - `training.loss.use_class_balanced: false`
  - `training.loss.use_focal_loss: false`
- **实验 E：Focal Loss**：
  - `training.loss.use_class_balanced: false`
  - `training.loss.use_focal_loss: true`
- **实验 F：MultiDatasetBalancedLoss（如后续启用）**：
  - 在代码中将 `cls_criterion` 切换为 `MultiDatasetBalancedLoss`，并使用 `dataset_ids` 对不同数据集加权。

对比要点：

- 少数类别（如某些情绪类别）的 F1 变化；
- 整体 F1 / Accuracy 是否因损失设计而显著提升；
- 对训练稳定性的影响（是否更易过拟合 / 欠拟合）。

### 6.3 实验命名与日志管理建议

- 建议在每次修改关键配置前，为 `config.experiment.name` 增加简短后缀，例如：  
  - `multimodal_emotion_recognition_baseline_noDA_standard`  
  - `multimodal_emotion_recognition_withDA_standard`  
  - `multimodal_emotion_recognition_withDA_emotionShift`
- 这样在 `logs/` 目录下，每次运行都会生成带时间戳和名称的子目录，方便在 TensorBoard 中对比不同配置的曲线。
- 对于后续论文中需要引用的关键实验，可以在本文件增加一小段“实验记录”（写清 run 名称、关键配置、最佳指标），相当于你自己的“混合数据集训练实验日记”。


