# Author: AI
# Date: 2026-03-10
# Description: 可直接用于毕业论文撰写的完整草稿（含实验策略与执行规范）

# 基于多源异构数据的配置驱动多模态情绪识别研究

## 摘要

多模态情绪识别是情感计算与智能驾驶人机交互的重要交叉方向。相比单模态识别，多模态方法能够联合利用文本语义、语音韵律、面部视觉线索及潜在生理特征，在复杂驾驶场景中提供更高的鲁棒性与可解释性。然而，面向真实工程落地的多模态研究仍面临三类关键挑战：其一，不同数据集之间存在显著域偏移与标注分布差异；其二，情绪类别天然长尾，类别不平衡导致模型倾向高频类别；其三，多模态输入质量波动显著，坏样本、缺失模态与训练恢复不规范会直接影响实验结论可信度。围绕上述问题，本文提出并实现了一套配置驱动的统一研究框架，以 `MultimodalEmotionModel` 为唯一主干，通过配置层控制模态组合、域适应（Domain Adaptation, DA）开关与融合策略切换，在不修改主干代码的前提下完成系统消融。

方法上，本文以 BERT、Wav2Vec2、ResNet50 作为三类主模态编码器，构建标准融合、情感转变感知融合、领导-跟随融合与两阶段融合等可切换模块；并结合功能最大相关思想（MFMC）与域对抗训练，探索跨数据集泛化能力提升路径。为避免“只看峰值、不看过程”的实验偏差，本文将工程可靠性作为方法学组成部分，建立了覆盖数据健康检查、断点续训一致性（含 scheduler 状态）、日志追踪与异常复盘的闭环流程。数据侧统一整合 CREMA-D、MELD、CMU-MOSEI，形成可复现训练入口与统一评估协议。

当前阶段已完成 T/A/V/AT/VT 模态消融、AT 有无 DA 对照、以及 from-pretrain 与 from-scratch 对照实验，并持续推进 AVT 相关实验。阶段性结果显示：文本模态在跨数据集混合训练中的稳定性最强；视频单模态表现较弱，但与文本组合后具有补偿增益；DA 与复杂融合策略并非在所有设置下稳定有效，必须在严格单变量控制下验证。本文最终贡献在于提出一条“统一骨架 + 配置驱动 + 工程闭环”的研究范式，将算法设计与可复现实验体系耦合，为后续论文复现、方法迭代与车载情绪感知应用提供可持续基础。

**关键词**：多模态情绪识别；跨数据集泛化；域适应；模态消融；配置驱动实验

---

## Abstract

Multimodal emotion recognition is important for intelligent interaction, driver behavior analysis, and human-computer collaboration. However, cross-dataset training suffers from domain shift, class imbalance, and heterogeneous modality quality. To address these issues, this thesis builds a configuration-driven multimodal emotion recognition framework using a single model backbone, `MultimodalEmotionModel`. Through controlled configuration switches, we systematically conduct modality, domain adaptation, and fusion-strategy ablations.

We unify three public datasets (CREMA-D, MELD, and CMU-MOSEI) and establish a reproducible engineering workflow including training, checkpoint resume, logging, and anomaly handling. Completed experiments cover modality ablations (T/A/V/AT/VT), AT domain adaptation comparison, and pretrain-then-finetune vs. scratch comparison. AVT experiments (with/without DA and with alternative fusion strategy) are also integrated into the planned protocol.

Current results show that text modality is relatively stable in mixed-dataset training. Video-only performance is weak, while video-text combinations provide complementary gains. Domain adaptation and advanced fusion strategies do not always guarantee consistent improvements and must be validated under strict controlled settings. By enforcing resume correctness (including scheduler state) and dataset health checks for corrupted media, we improve reliability and interpretability of experimental conclusions. The thesis contributes a practical and reproducible pipeline: single-backbone modeling, configuration-driven ablation, and engineering-level reliability assurance.

**Keywords**: multimodal emotion recognition; cross-dataset generalization; domain adaptation; ablation study; configuration-driven experiments

---

## 第1章 绪论

本章围绕研究动机、问题定义与论文贡献展开，先说明智能驾驶场景下多模态情绪识别的现实需求，再明确本文的研究边界、核心问题与技术路线定位，为后续相关研究分析与方法设计奠定基础。

### 1.1 研究背景与意义

随着智能汽车从 L2/L2+ 辅助驾驶向 L3/L4 条件自动驾驶演进，车内人机协同关系正在从“指令响应”转向“状态感知 + 主动交互”[18]。在这一过渡阶段，驾驶员仍需在关键时刻完成接管，因此系统不仅要理解外部交通环境，还要实时理解驾驶员内部状态[18]。传统驾驶员监测系统（DMS）主要关注闭眼时长、头姿与注意力分散，能够覆盖疲劳与分心，但对路怒、焦虑、压抑等情绪状态感知能力有限，难以支撑下一代情感化智能座舱[17][18]。

情感计算理论指出，情绪状态会显著影响认知负荷、风险偏好与操作稳定性[1]。驾驶场景中，愤怒可能引发激进操作，焦虑可能导致过度谨慎或反应迟缓，持续负性情绪还会降低接管质量[16][17]。因此，面向驾驶场景开展多模态情绪识别研究具有双重价值：一方面可为安全干预提供前置指标，另一方面可为智能座舱个性化交互提供高价值上下文[16][18]。

与通用多媒体情感任务不同，驾驶场景具有“强时序、强噪声、强异构”的数据特征：光照变化、遮挡和噪声会导致视频与音频质量波动；语音转文本存在识别误差；不同数据集的标签体系与采样协议差异明显。上述因素导致跨数据集训练易出现域偏移、类别不均衡和实验不稳定。基于此，本文将“方法性能”与“工程可信度”统一纳入研究目标，强调可复现、可对照、可解释的研究路径。

### 1.2 研究问题

围绕本课题，本文聚焦以下问题：

1. 在统一模型骨架下，不同模态组合（T/A/V/AT/VT/AVT）对性能的贡献规律是什么？
2. 域适应（DA）是否能在跨数据集混合训练中稳定提升性能？
3. 融合策略从 `standard` 切换到 `emotion_shift` 是否带来可重复收益？
4. 预训练+微调相较于从零训练是否具有稳定优势？
5. 如何通过工程策略保障实验结果可复现、可解释？

进一步地，本文补充两个与论文写作和工程一致性强相关的问题：

6. 在文本建模上，BERT 与传统 Word2Vec 路线的能力边界分别是什么，为什么本项目选择 BERT 作为主线编码器？
7. 在多模态方法比较中，如何将论文中引用的方法（CFN-ESA、MFMC、Leader-Follower、GA2MIF）与现有代码模块形成一一映射，从而保证“文献-方法-实验”证据链闭环？

### 1.3 研究内容与贡献

本文的主要工作与贡献如下：

1. **统一模型骨架与配置驱动实验机制**  
   基于 `MultimodalEmotionModel` 构建统一实验框架，通过 YAML 配置控制模态、DA、融合策略，避免“每个实验改代码”的不可控风险。

2. **多源数据统一与跨数据集训练实践**  
   对 CREMA-D、MELD、CMU-MOSEI 数据进行统一组织与标签映射，建立混合训练流程并配套日志标准。

3. **系统化消融实验设计**  
   从模态、DA、融合策略、训练范式四个层级建立实验矩阵，形成可直接用于论文撰写的证据链结构。

4. **工程可靠性闭环**  
   针对续训断崖与坏视频拖慢问题，形成恢复规范、健康检查与异常复盘机制，提高结论可信度。

5. **文献方法与工程实现对齐**  
   将核心参考工作映射至项目模块：`models/emotion_shift.py`（CFN-ESA思想）、`models/functional_correlation.py`（MFMC思想）、`models/leader_follower_attention.py`（Leader-Follower思想）、`models/two_stage_fusion.py`（GA2MIF思想），保证论文叙述与代码实现一致。

6. **文本建模路线阐释**  
   在论文中系统比较 Word2Vec 与 BERT：Word2Vec提供静态词向量、成本低且可解释；BERT提供双向上下文表征、可建模多义词与长距离依赖。结合本项目跨语境与跨数据集需求，最终采用 BERT 作为主线，并将 Word2Vec 作为理论对照与轻量基线参考。

### 1.4 论文结构

本文其余章节安排如下：第2章介绍相关研究；第3章阐述模型与方法；第4章给出实验设置与结果分析；第5章总结与展望。

### 1.5 本章小结

本章从智能驾驶人机协同需求出发，论证了多模态情绪识别在安全与交互两方面的研究价值；在此基础上，明确了本文围绕模态组合、域适应、融合策略、训练范式与工程可复现性开展研究。通过给出研究贡献与章节结构，本文完成了整体研究问题的框架化定义。

---

## 第2章 相关技术与研究现状

本章从“任务属性—方法演进—关键瓶颈—本文定位”四个层面展开。首先，明确驾驶场景下多模态情绪识别与通用情感分析任务的差异；其次，系统梳理多模态融合方法从浅层融合到动态融合的发展脉络；随后，结合跨数据集训练场景讨论域偏移与类别不平衡问题；最后，基于现有研究不足给出本文的技术切入点与研究必要性。

### 2.1 多模态情绪识别任务内涵与场景特征

多模态情绪识别（MER）旨在联合视觉、语音、文本及可选生理信号，推断离散情绪类别或连续情绪维度。与通用场景相比，驾驶场景在数据与任务上具有以下特点：

1. **情绪动态性强**：驾驶员情绪受路况、交互事件、认知负荷持续影响，呈现明显时序演化而非静态状态；
2. **模态质量波动大**：低光、遮挡、车内噪声会导致视觉与语音信号质量不稳定；
3. **语义与行为耦合**：文本语义（如自发语言）与驾驶行为风险存在强关联；
4. **跨域差异显著**：不同数据集在采样协议、标签体系、语言风格和采集环境方面差异明显。

因此，驾驶场景的 MER 不仅是“更高准确率”的问题，更是“稳定性、泛化性、可解释性、可复现性”的综合优化问题[16][17][18]。

### 2.2 相关研究的技术演进路径

从方法演进看，MER 大致经历三个阶段：

#### 2.2.1 第一阶段：浅层融合与独立建模

早期方法多采用单模态独立编码后进行特征拼接或决策加权。该路线实现简单，但存在两个缺点：  
- 难以表达“模态间依赖关系”；  
- 对时序动态变化敏感度不足。  
在驾驶情绪快速波动场景中，这种方法往往只能得到“平均化情绪”，难以捕捉突变。

#### 2.2.2 第二阶段：跨模态注意力与Transformer范式

随着 Transformer 在序列建模中的成功，跨模态注意力成为主流。MulT 等工作通过跨模态交互学习未对齐序列间的依赖关系，显著提升了融合能力[7]。这类方法相比浅层融合的优势在于：

1. 可以在特征层面动态建权；
2. 对长程依赖更敏感；
3. 可统一处理不同长度序列。

但该类方法也面临“计算成本较高、训练对数据质量敏感”的现实问题。

#### 2.2.3 第三阶段：动态融合与鲁棒增强

近年方法进一步关注“情绪转变”和“模态质量不均”：

- CFN-ESA 强调情绪转移建模，通过转变感知增强动态情绪识别能力[8]；
- Leader-Follower 通过非对称引导缓解低质量模态失效问题[10]；
- GA2MIF 通过图建模 + 两阶段融合增强结构化关系表达[9]；
- MFMC 从相关性学习角度提升跨模态表示一致性[11]。

这些工作共同推动 MER 从“静态融合”迈向“动态、鲁棒、可迁移融合”。

### 2.3 多模态融合策略比较

#### 2.3.1 早期融合、后期融合与中间融合

1. **早期融合**：在输入或浅层特征阶段拼接，优势是实现简单，缺点是对齐要求高、噪声传播明显；  
2. **后期融合**：在分类器输出层融合，优势是鲁棒，缺点是跨模态细粒度关系利用不足；  
3. **中间融合**：在中间表示层进行交互，通常结合注意力、图网络、门控机制，是当前性能与可解释性较均衡的路径[7][19]。

#### 2.3.2 本研究采用的四类融合思想

本文围绕统一主干，设置四类可切换融合策略，用于严格消融：

1. **标准融合**：作为稳定基线，用于提供对照参照系；
2. **情绪转变感知融合**：强调时序变化幅度，适配突发情绪事件；
3. **领导-跟随融合**：在模态质量不均时提升低质量模态可用性；
4. **两阶段融合**：先上下文关系建模，再进行跨模态决策融合。

这种设计并非追求“所有策略都最优”，而是追求“在同一训练骨架下可公平比较”。

### 2.4 驾驶场景中的跨域泛化问题

跨数据集训练是本文实验主线之一。域偏移通常由以下因素引发：

1. **采集域差异**：设备、采样率、画质、噪声条件不同；
2. **语义域差异**：文本语言风格、语句长度、ASR误差模式不同；
3. **标签域差异**：类别定义边界、标注主观性、类别分布差异；
4. **任务域差异**：对话情绪与驾驶情绪触发机制不同。

在该背景下，域适应并非“默认有效”，其收益高度依赖模态组合与训练设置。本文通过 noDA/DA 严格对照，避免先验假设直接替代实验结论。

### 2.5 类别不平衡与评价偏差问题

MER 常见长尾分布：中性样本比例高，风险情绪比例低。若仅报告 Accuracy，容易掩盖少数类识别不足。为降低评价偏差，本文强调：

1. 同时报告 `Best` 与 `Last`，避免只看峰值；
2. 强调 F1 指标，关注少数类识别；
3. 在可能条件下补充重复实验，报告均值与方差；
4. 将“训练稳定性”纳入结论解释，而不仅是单点性能。

### 2.6 文本表示方法：Word2Vec 与 BERT 的边界

#### 2.6.1 Word2Vec 的价值与局限

Word2Vec 通过上下文预测学习静态词向量，计算效率高、部署成本低、可解释性较强[3]。但其核心局限在于“同词同向量”，对语境变化不敏感，难以建模情绪语义中的隐含转折与上下文反讽。

#### 2.6.2 BERT 的优势与代价

BERT 基于双向上下文建模，能够捕捉语义依赖、词义歧义与长距离关系[2][6]。在跨语境、跨数据集任务中更具表达能力，但也带来更高训练开销与数据质量要求。

#### 2.6.3 本文选择依据

结合本研究“跨数据集 + 多模态对齐 + 动态情绪识别”的目标，选择 BERT 作为主线文本编码器，并以 Word2Vec 作为理论对照。该决策本质上是“表达能力优先”的研究取舍。

### 2.7 驾驶情绪研究与多模态方法的结合趋势

智能驾驶研究已从“外部目标感知”逐步走向“车内状态感知”。驾驶员情绪不仅影响交互体验，也影响行为风险水平。相关研究表明，情绪状态可显著影响轨迹与决策行为[16][17]，这为将 MER 引入智能座舱提供了应用依据。

从发展趋势看，未来研究将更关注：

1. 多任务联合（情绪 + 意图 + 风险）；
2. 跨域迁移与持续学习；
3. 工程可复现性与可解释性并重；
4. 学术指标与部署成本协同优化。

### 2.8 本章小结

本章系统梳理了 MER 在驾驶场景下的任务属性、方法演进与关键挑战。总体上，现有方法在融合能力上持续提升，但在跨域稳定性与实验可复现性方面仍存在不足。本文据此确立“统一骨架、配置驱动、严格对照、工程闭环”的研究路径，并在后续章节展开具体方法与实验验证。

---

## 第3章 方法设计

本章详细阐述本文提出的配置驱动多模态情绪识别方法。与常见“模型即方案”不同，本文方法由“统一模型骨架 + 可切换策略模块 + 规范化训练流程”共同构成，其目标不仅是提升性能，更是保证跨实验可比性与结果可复现性。

### 3.1 研究方法总体思路

本文方法由四个核心层次组成：

1. **多模态输入层**：接收视频、音频、文本（并预留生理信号接口）；
2. **模态编码层**：分别提取各模态高层语义表示；
3. **融合决策层**：根据设定策略进行跨模态信息交互与加权；
4. **任务优化层**：在分类主任务基础上，引入可选域对齐与相关性约束。

该设计的关键原则是“主骨架稳定、变量显式外置”。即：任何实验对照应优先通过配置变量变化完成，尽量避免改动核心前向逻辑，从源头降低实验耦合。

### 3.2 多模态输入建模与数据表示

#### 3.2.1 视频模态

视频模态主要承载表情、头部姿态、微动作等视觉线索。输入可来自原始帧序列或预提取特征，经过视觉编码器得到固定维度表示。考虑到驾驶场景中存在遮挡与低照条件，视频模态在本文中既作为独立信息源，也作为与文本互补的信息补充源。

#### 3.2.2 音频模态

音频模态承载韵律、语速、音强变化等情绪线索。本文采用端到端预训练语音编码方式，避免传统手工特征在复杂噪声环境下表达受限的问题。音频模态在路怒、焦虑等情绪状态的动态检测中具有较高敏感性。

#### 3.2.3 文本模态

文本模态承载语义和情绪意图信息，是本研究中稳定性较高的模态。文本输入来源可为标注文本或语音识别结果。通过上下文编码后，文本表示在跨数据集训练中起到“语义锚点”作用，可缓解视觉与音频质量波动对最终决策的影响。

#### 3.2.4 生理信号模态（扩展接口）

本文主实验以音视频文本三模态为主，同时保留生理模态扩展能力，以便后续接入 EEG/GSR 等信号进行联合建模。该设计保障了方法在未来驾驶员状态综合感知任务中的扩展性。

### 3.3 模态编码器选择依据与作用机理

#### 3.3.1 视觉编码：ResNet50

ResNet50 通过残差结构缓解深层网络优化难题，在视觉迁移学习任务中稳定性高[4]。其在本文中的角色是提供高层视觉语义特征，并作为可复现实验基线的视觉主干。选择该主干的原因包括：成熟稳定、社区验证充分、工程成本可控。

#### 3.3.2 语音编码：Wav2Vec2

Wav2Vec2 通过自监督学习获得高质量语音表示，能够在弱标注场景中保持较强泛化能力[5]。对于驾驶舱噪声条件下的语音建模，该编码方式相较手工声学特征具有更强鲁棒性。

#### 3.3.3 文本编码：BERT

BERT 具备双向上下文建模能力，对情绪语义中常见的语境依赖和隐式表达具有更好适配性[2][6]。相较 Word2Vec 静态词向量，BERT 在跨语境迁移和多义词建模方面表现更优，因此作为本文主线文本编码方法。

#### 3.3.4 跨模态统一表示

各模态编码后需投影到统一表示空间，目的是：

1. 降低不同编码器输出尺度不一致带来的融合偏差；
2. 使后续融合策略可在同一维度空间下公平比较；
3. 便于扩展新模态时保持接口一致。

### 3.4 融合策略设计与理论动机

本文设置四类融合策略进行对照：

#### 3.4.1 标准融合（Standard）

作为实验基线，主要用于提供稳定对照。其优势是结构简洁、训练稳定，适合作为“默认参照系”。

#### 3.4.2 情绪转变感知融合（Emotion Shift）

该策略借鉴 CFN-ESA 思想[8]，核心是将“情绪变化强度”显式引入融合决策。对于驾驶场景中的突发事件（如急刹、加塞触发情绪波动），该策略能够提升模型对短时情绪跃迁的敏感度。

#### 3.4.3 领导-跟随融合（Leader-Follower）

该策略借鉴 Leader-Follower 非对称交互思想[10]。当某一模态质量较高时，其表示用于引导其他模态聚焦关键区域，从而缓解“弱模态拖累”问题，提升噪声场景鲁棒性。

#### 3.4.4 两阶段融合（Two-Stage）

借鉴 GA2MIF 思路[9]，先建模上下文关系，再建模跨模态交互。该策略适用于关系结构更复杂的场景，能够更细粒度地表达模态内与模态间依赖。

### 3.5 域适应与相关性约束机制

#### 3.5.1 域适应机制

针对跨数据集分布差异，本文设置可选域适应分支。其目标是通过对抗式特征对齐降低数据域差异影响，使模型学习到更具域不变性的表示。需要强调的是，域适应并非在所有设置下都带来增益，因此必须通过严格对照验证其有效性。

#### 3.5.2 相关性约束机制

借鉴 MFMC 思想[11]，本文引入可选相关性约束项，鼓励不同模态在高层语义上保持一致性。该约束的理论动机是：情绪属于跨模态共享语义，不同模态应在抽象表示层面存在可学习相关结构。

### 3.6 统一损失函数设计

本文采用可组合损失函数框架：

\[
\mathcal{L} = \lambda_{cls}\mathcal{L}_{cls} + \lambda_{reg}\mathcal{L}_{reg} + \lambda_{trend}\mathcal{L}_{trend} + \lambda_{da}\mathcal{L}_{da} + \lambda_{corr}\mathcal{L}_{corr}
\]

其中，\(\mathcal{L}_{cls}\) 为主任务分类损失；\(\mathcal{L}_{reg}\) 与 \(\mathcal{L}_{trend}\) 分别用于可选回归和趋势学习；\(\mathcal{L}_{da}\) 用于域对齐；\(\mathcal{L}_{corr}\) 用于跨模态相关性增强。通过权重系数控制启停，可在统一训练框架下完成多种实验设定。

### 3.7 训练流程与实验可复现机制

本文训练流程遵循“统一入口、统一日志、统一恢复”的规范：

1. 统一入口执行预训练与微调；
2. 每轮训练输出结构化指标与可视化记录；
3. 断点恢复必须同时恢复模型参数、优化器状态、学习率调度状态；
4. 恢复后检查曲线连续性，确保实验可比；
5. 训练前完成媒体健康检查，避免坏样本影响收敛行为。

该规范将“工程流程正确性”提升为方法学的一部分。

### 3.8 配置驱动实验方法学

本文所有关键变量通过配置层管理，包括：

1. 模态组合变量（T/A/V/AT/VT/AVT）；
2. 融合策略变量（standard/emotion_shift/leader-follower/two-stage）；
3. 域适应变量（noDA/DA）；
4. 训练范式变量（from-pretrain/from-scratch）；
5. 损失与超参数变量（是否启用平衡损失、相关性约束、域损失权重等）。

配置驱动的核心意义在于：将“实验定义”从代码实现中解耦，形成结构化实验矩阵，便于后续复现与审查。

### 3.9 算法复杂度与计算代价分析

设批量大小为 \(B\)、时间长度为 \(T\)、隐藏维度为 \(d\)、模态数为 \(M\)：

1. 编码阶段：Transformer类编码近似 \(O(B\cdot T^2\cdot d)\)，视觉卷积编码受分辨率与卷积核规模影响；
2. 融合阶段：若进行全对全模态交互，复杂度近似 \(O(B\cdot M^2\cdot T^2\cdot d)\)；
3. 动态建模阶段：时序分支复杂度一般低于全注意力主项；
4. 域适应阶段：额外引入对抗分支，计算代价主要体现为训练时间增量。

实际训练中，耗时瓶颈通常来自视频 I/O 与序列编码，而非分类头本身。

### 3.10 方法与文献映射关系

为保证学术表述与实现逻辑一致，本文将关键方法来源明确映射为：

1. 情绪转变感知融合 <- CFN-ESA[8]；
2. 相关性增强约束 <- MFMC[11]；
3. 非对称模态引导 <- Leader-Follower[10]；
4. 两阶段关系融合 <- GA2MIF[9]；
5. 文本、语音、视觉编码 <- BERT[2]、Wav2Vec2[5]、ResNet[4]。

该映射避免“论文写法与实现逻辑脱节”，提升答辩可追溯性。

### 3.11 本章小结

本章提出并详细阐述了本文方法：在统一骨架下，通过可切换融合策略、可组合损失函数与配置驱动实验机制实现系统化研究；通过训练恢复规范与数据健康检查保障实验可复现性；并通过文献映射确保方法来源清晰、论证链条完整。下一章将在此基础上给出实验设计与结果分析。

---

## 第4章 实验设置与结果分析

本章从实验可复现性出发，系统给出实验环境、数据组织、对照矩阵与阶段结果，并结合工程日志分析方法有效性与稳定性。为满足学位论文提交要求，本章同时提供图表编号规范与终稿回填模板。

### 4.1 实验环境与实现细节

本研究所有实验在统一工程框架下完成，保证实验之间具有可比性与可复现性。实验平台采用 Linux 服务器环境，训练框架为 PyTorch，预训练特征提取主干分别为 ResNet50（视频）、Wav2Vec2（音频）与 BERT（文本）。统一训练入口为 `scripts/train.py`，实验配置通过 `config/*.yaml` 管理。训练过程采用 TensorBoard 与 `metrics.csv` 双重记录机制，分别用于可视化分析与结构化统计。

### 4.2 数据集与统一预处理

实验使用 CREMA-D、MELD 与 CMU-MOSEI 三个公开数据集。为降低工程复杂度并统一训练接口，本文将数据整理为统一目录结构：

`data/train|val|test/{video,audio,text,labels}`

在正式训练前，统一执行数据健康检查命令：

`python scripts/check_media_health_dir.py --data_dir data`

该步骤用于识别损坏媒体文件与不可读取样本，避免异常样本对训练耗时、收敛行为及结果解释造成干扰。

### 4.3 实验变量与对照原则

#### 4.3.1 变量定义

1. 模态变量：T/A/V/AT/VT/AVT；
2. DA变量：noDA vs DA；
3. 融合变量：standard vs emotion_shift（后续可扩展）；
4. 训练范式变量：from-pretrain vs from-scratch。

#### 4.3.2 对照原则

为保证结论有效性，本文遵循“单变量控制”原则：每次消融仅改变一个主变量，其余设置保持一致（包括学习率、epoch、batch size、数据划分与日志策略）。该原则用于确保实验结果可解释且具备统计比较意义。

### 4.4 已完成实验概览（基于当前工程进度）

1. 模态实验：T、A、V、AT、VT（已完成）；
2. 域适应实验：AT_noDA vs AT_DA（已完成）；
3. 训练范式：CREMA from-pretrain vs from-scratch（已完成）；
4. 工程异常复盘：  
   - 续训曲线断崖（scheduler 未恢复）已修复；  
   - 坏视频导致训练慢问题已建立排查流程。
5. AVT 进度：  
   - `AVT_pretrain_3datasets_noDA_20260323_202809` 已启动（当前日志尚未形成可用 val 指标，按处于进行阶段处理）。

> 说明：上述实验结果以文档 `PROJECT_OVERVIEW_AND_TRAINING_PLAN.md` 的第10章日志与对应 `logs/*/metrics.csv` 为准。

### 4.5 待完成实验（按论文主线）

1. `AVT_noDA`：`config/config_AVT_noDA.yaml`
2. `AVT_DA`：`config/config_AVT_DA.yaml`
3. `AVT_noDA_emotion_shift`：`config/config_AVT_noDA_emotion_shift.yaml`

### 4.6 实验顺序（严格执行版）

#### 阶段 P0（已完成）

- 固化现有结果；
- 统一 run 命名；
- 补齐异常复盘。

#### 阶段 P1（立即执行）

1. AVT_noDA
2. AVT_DA
3. AVT_noDA_emotion_shift

#### 阶段 P2（形成论文主表）

1. 模态消融表：T/A/V/AT/VT/AVT
2. DA消融表：AT、VT、AVT 的 noDA vs DA
3. 融合消融表：AVT_standard vs AVT_emotion_shift

#### 阶段 P3（增强证据）

- 补 MELD 或 MOSEI 的 from-pretrain vs from-scratch；
- 关键组多次重复（可选）统计均值和标准差。

### 4.7 结果记录模板（已自动回填已完成实验）

#### 表4-1 主结果总览（终稿回填模板）

| 实验名 | 模态 | 融合 | DA | 训练范式 | Best Acc | Best F1 | Last Acc | Last F1 | 总时长 | 状态 | 结论 |
|---|---|---|---|---|---:|---:|---:|---:|---|---|---|
| AT_pretrain_3datasets_noDA_20260305 | AT | standard | noDA | pretrain | 0.1544 | 0.2675 | 0.1361 | 0.1833 | 约 8.35 hr | Completed | 已完成，作为 AT noDA 基线 |
| VT_pretrain_3datasets_noDA_20260318_203248 + VT_pretrain_3datasets_noDA_20260321_142141 | VT | standard | noDA | pretrain(续训合并) | 0.1166 | 0.1889 | 0.0821 | 0.1218 | 约 2.49 day | Completed | 已完成，存在后期回落 |
| AVT_pretrain_3datasets_noDA_20260323_202809 | AVT | standard | noDA | pretrain | 待回填 | 待回填 | 待回填 | 待回填 | 待回填 | Running | 当前未完成 |
| AVT_pretrain_3datasets_DA_YYYYMMDD_HHMMSS | AVT | standard | DA | pretrain | 待回填 | 待回填 | 待回填 | 待回填 | 待回填 | Planned | 待执行 |
| AVT_pretrain_3datasets_noDA_emotion_shift_YYYYMMDD_HHMMSS | AVT | emotion_shift | noDA | pretrain | 待回填 | 待回填 | 待回填 | 待回填 | 待回填 | Planned | 待执行 |
| AT_crema_from_pretrain_noDA_20260313_212316 | AT | standard | noDA | finetune(from-pretrain, CREMA) | 0.1788 | 0.3033 | 0.1761 | 0.2974 | 约 1.69 hr | Completed | 已完成，微调效果稳定 |
| AT_crema_scratch_noDA_20260314_102323 | AT | standard | noDA | finetune(from-scratch, CREMA) | 0.1747 | 0.2975 | 0.1882 | 0.2275 | 约 31.7 min | Completed | 已完成，终点低于 from-pretrain |

#### 表4-2 模态消融（终稿回填模板）

| 模态组合 | use_video | use_audio | use_text | Best Acc | Best F1 | Last Acc | Last F1 | 状态 | 结论 |
|---|---|---|---|---:|---:|---:|---:|---|---|
| T | ✗ | ✗ | ✓ | 0.2154 | 0.2734 | 0.1609 | 0.1761 | Completed | 已完成，当前最稳单模态 |
| A | ✗ | ✓ | ✗ | 0.1544 | 0.2675 | 0.0810 | 0.1499 | Completed | 已完成，后期退化明显 |
| V | ✓ | ✗ | ✗ | 0.0751 | 0.1396 | 0.0864 | 0.1285 | Completed | 已完成，单视频偏弱 |
| AT | ✗ | ✓ | ✓ | 0.1544 | 0.2675 | 0.1361 | 0.1833 | Completed | 已完成，稳定优于 V |
| VT | ✓ | ✗ | ✓ | 0.1166 | 0.1889 | 0.0821 | 0.1218 | Completed | 已完成，有峰值收益但后期回落 |
| AVT | ✓ | ✓ | ✓ | 待回填 | 待回填 | 待回填 | 待回填 | Running | 处于进行阶段 |

#### 表4-3 DA 消融（终稿回填模板）

| 模态 | noDA-Best F1 | DA-Best F1 | noDA-Last F1 | DA-Last F1 | 差值（Best） | 状态 | 结论 |
|---|---:|---:|---:|---:|---:|---|---|
| AT | 0.2675 | 0.2584 | 0.1833 | 0.1384 | -0.0091 | Completed | 已完成：DA 未带来稳定提升 |
| VT | 0.1889 | 待回填 | 0.1218 | 待回填 | 待回填 | Planned | VT_DA 待补实验 |
| AVT | 待回填 | 待回填 | 待回填 | 待回填 | 待回填 | Planned | AVT noDA/DA 待形成对照 |

#### 表4-4 融合策略消融（固定 AVT，终稿回填模板）

| 模态 | 策略 | DA | Best Acc | Best F1 | Last Acc | Last F1 | 总时长 | 状态 | 结论 |
|---|---|---|---:|---:|---:|---:|---|---|---|
| AVT | standard | noDA | 待回填 | 待回填 | 待回填 | 待回填 | 待回填 | Running | `AVT_pretrain_3datasets_noDA_20260323_202809` |
| AVT | emotion_shift | noDA | 待回填 | 待回填 | 待回填 | 待回填 | 待回填 | Planned | 使用 `config/config_AVT_noDA_emotion_shift.yaml` |

### 4.8 图表编排建议（含正文引用模板）

1. **图4-1 训练损失曲线对比图**：`train/loss_classification`（AT/VT/AVT）  
   - 正文引用模板：如图4-1所示，不同模态组合在训练阶段呈现出明显不同的收敛轨迹，其中 AT 收敛较为平稳，VT 在中后期出现波动加剧现象。  
2. **图4-2 验证F1曲线对比图**：`val/f1`（关键组对比）  
   - 正文引用模板：由图4-2可见，VT 在中期存在阶段性提升，但终点未能保持同等优势，提示其泛化稳定性仍需进一步优化。  
3. **图4-3 断点续训修复对比图**：恢复前后曲线连续性案例  
   - 正文引用模板：图4-3显示，在恢复 scheduler 状态后，续训曲线连续性显著改善，验证了恢复策略修复的有效性。  
4. **图4-4 数据清洗与训练效率关系图（可选）**：坏样本处理前后耗时对比  
   - 正文引用模板：图4-4表明，坏样本清洗可降低数据加载异常频率，从而改善整体训练效率。

### 4.9 结果分析写作模板（可直接替换数字）

#### 4.9.1 模态贡献分析

在统一骨架与相同训练框架下，文本模态在混合数据集场景中表现出更强稳定性。视频单模态在当前输入规模下表现较弱，但视频与文本组合后出现明显互补，说明视频信息具有增益潜力但需依赖其他模态提供语义锚点。

#### 4.9.2 DA 有效性分析

DA 在不同模态组合中的效果存在差异。若 DA 仅带来偶然峰值而未改善终点指标，应判定其收益不稳定；若在多次实验中 Best 与 Last 指标均持续改善，才可认为 DA 在该设置下有效。

#### 4.9.3 融合策略分析

高级融合策略（如 emotion_shift）只有在可重复实验中持续优于 standard，且训练成本可接受时，才具备工程推广价值。若收益有限或波动增加，standard 仍是更优实践基线。

#### 4.9.4 工程可靠性分析

续训流程中恢复学习率调度器状态对曲线连续性至关重要。坏视频样本会显著影响训练效率与稳定性。工程规范（tmux、checkpoint、health check）直接决定实验结果可信度。

### 4.10 准提交版结论段（已按当前结果自动生成，可直接粘贴正文）

#### 4.10.1 主结果结论（对应表4-1）

在三数据集混合预训练场景下，AT_noDA 作为稳定基线取得了 `Best F1=0.2675`、`Last F1=0.1833`。VT_noDA（两段续训合并）取得 `Best F1=0.1889`、`Last F1=0.1218`，表现出“中期有提升峰值、后期存在回落”的特征。当前 AVT_noDA 仍处于训练阶段，AVT_DA 与 AVT_noDA_emotion_shift 尚待执行，因此三模态最终结论需在 AVT 系列实验闭环后给出。  

在单数据集微调（CREMA）中，from-pretrain 与 from-scratch 的对比显示：from-pretrain 的结果更稳定（`Best F1=0.3033`，`Last F1=0.2974`），而 from-scratch 虽在早期可达到相近峰值（`Best F1=0.2975`），但终点回落更明显（`Last F1=0.2275`），说明预训练初始化对稳定收敛具有积极作用。

#### 4.10.2 模态消融结论（对应表4-2）

已完成实验显示，文本单模态 T 的稳定性最强（`Best F1=0.2734`），音频单模态 A 在后期退化明显（`Last F1=0.1499`），视频单模态 V 整体偏弱（`Best F1=0.1396`）。  
双模态中，AT 的综合表现优于 VT 与 V，说明在当前训练配置下“文本+音频”仍是更稳的组合；VT 虽达到 `Best F1=0.1889`，但终点回落至 `0.1218`，提示视频模态在现有输入规格与数据质量条件下仍有稳定性瓶颈。  

#### 4.10.3 DA消融结论（对应表4-3）

在 AT 场景中，DA 未带来稳定增益：`noDA-Best F1=0.2675` 高于 `DA-Best F1=0.2584`，且终点指标 `noDA-Last F1=0.1833` 亦高于 `DA-Last F1=0.1384`。这一结果说明“是否启用 DA”与模态组合、数据状态和训练超参数高度相关，不能假设其必然提升性能。  
VT 与 AVT 的 DA 对照尚未完成，后续需以相同训练配方补齐对照后再给出最终 DA 结论。

#### 4.10.4 融合策略结论（对应表4-4）

当前可确认的事实是：`standard` 融合策略已支持完整实验链路并可稳定产生日志结果；`emotion_shift` 的 AVT 对照尚未执行，因此“高级融合是否优于 standard”暂不下最终结论。论文写作中应将该结论明确标注为“阶段性结论”，避免过度推断。

#### 4.10.5 本阶段可写入论文的小结（可直接用）

综合当前结果，在统一模型骨架下，配置驱动消融能够有效揭示模态贡献规律：文本模态在跨数据集场景中稳定性较高，视频模态具有潜在互补价值但当前稳定性不足。AT 的 DA 对照未显示明确收益，提示域适应策略需要在更严格条件下进一步验证。预训练+微调相较于从零训练具有更好的终点稳定性。后续工作的关键是完成 AVT 系列（noDA/DA/emotion_shift）并形成三模态闭环证据，以支撑最终论文结论的完整性。

### 4.11 项目代码结构与系统分层设计

为确保方法研究与工程实现协同演进，本文采用“分层解耦 + 接口统一 + 配置驱动”的系统结构。整体可抽象为五层：

1. **数据层**：完成多源样本组织、模态对齐与质量控制；
2. **表示层**：完成视觉、语音、文本等模态编码；
3. **融合层**：根据策略执行跨模态交互与信息聚合；
4. **任务层**：输出分类主任务与可选辅助任务；
5. **实验控制层**：统一管理训练参数、实验变量与日志规范。

该结构的核心意义在于：当研究变量变化时，优先调整实验配置而不是改动主干流程，从而减少隐式变量引入，保证对照实验可信度。

### 4.12 模型层详细实现（与论文方法一一对应）

#### 4.12.1 主模型装配逻辑

统一主模型承担如下职责：

- 初始化多模态编码分支；
- 按融合策略执行跨模态交互；
- 输出分类主任务结果，并按需输出辅助分支结果；
- 在同一主干下支持多种模态组合实验。

该设计保证了实验对照中的“唯一变量可控”。

#### 4.12.2 特征提取器设计

本研究采用视觉、语音、文本三类主编码器，并预留生理信号扩展接口。编码后特征统一投影到同维空间，以降低不同模态在尺度和分布上的偏差。

各分支输出后投影到统一维度，避免融合层出现维度耦合。

#### 4.12.3 融合策略模块化

融合机制采用可插拔设计，以便进行公平消融比较。策略包括标准融合、情绪转变感知融合、领导-跟随融合与两阶段融合。各策略在同一训练框架中切换，仅改变融合机制本身，从而保证结论可归因。

#### 4.12.4 域适应与分布对齐

域适应模块基于对抗式对齐思想，通过域判别学习与特征反向对齐降低跨数据集分布差异。该机制在实验中作为独立开关控制，确保 noDA/DA 对照的可重复与可解释。

#### 4.12.5 不平衡损失与相关性约束

针对类别长尾与跨模态一致性问题，本文采用可选的平衡损失与相关性约束项。前者提升少数类学习能力，后者提升模态间语义一致性。两类机制均作为可控变量纳入实验矩阵。

### 4.13 脚本层与实验执行链路

#### 4.13.1 训练主流程

训练流程采用统一入口，核心能力包括：

1. 预训练与微调阶段切换；
2. 断点续训；
3. 多数据集训练模式下的日志命名与指标落盘；
4. 每 epoch 写入 CSV/JSON/TensorBoard；
5. 保存最佳模型与周期 checkpoint。

核心流程可概括为：加载配置 -> 构建模型与数据加载器 -> 训练与验证循环 -> 指标记录 -> 最优模型更新 -> checkpoint 存档。

#### 4.13.2 推理与基线流程

系统提供标准化推理流程与批量基线执行流程，保证训练结果可复核、可比较、可复现实验重跑。

#### 4.13.3 数据治理脚本链

数据工程遵循“采集/下载—整理—统一—质检—治理”闭环。该流程将坏样本治理前置到训练前，显著降低训练中断与异常波动风险，是本文实验可复现性的关键保障。

### 4.14 配置驱动实验矩阵（全量映射）

本文采用结构化配置管理实验变量。配置矩阵的核心目标是把“实验定义”从代码实现中抽离，使模态组合、域适应开关、融合策略与训练范式均可通过参数层统一控制。主要配置与研究问题的映射关系如下：

| 配置文件 | 研究目的 | 关键变量 |
|---|---|---|
| `config_text_only.yaml` | 文本单模态基线 | `use_text=true` |
| `config_audio_only.yaml` | 音频单模态基线 | `use_audio=true` |
| `config_video_only.yaml` | 视频单模态基线 | `use_video=true` |
| `config_VT_noDA.yaml` | 双模态互补验证（VT） | `use_video=true,use_text=true,DA=false` |
| `config_AT_DA.yaml` | AT 域适应对照 | `AT + DA=true` |
| `config_AVT_noDA.yaml` | 三模态标准融合基线 | `AVT + standard + DA=false` |
| `config_AVT_DA.yaml` | 三模态域适应验证 | `AVT + standard + DA=true` |
| `config_AVT_noDA_emotion_shift.yaml` | 融合策略对照 | `AVT + emotion_shift + DA=false` |
| `config_crema_finetune_from_pretrain.yaml` | 预训练迁移收益验证 | `finetune from-pretrain` |
| `config_crema_finetune_from_scratch.yaml` | 从零训练对照 | `finetune from-scratch` |

通过该矩阵可直接生成论文中的“实验设计总表”，并保证后续审稿/答辩时可追溯。

### 4.15 训练过程、异常处理与复现实践

#### 4.15.1 训练过程中的关键规范

1. 所有实验均采用统一训练入口；
2. 所有实验产出结构化指标文件（`metrics.csv`）；
3. 所有关键训练过程写入 TensorBoard；
4. 同一类消融严格保持非目标变量不变；
5. 实验命名包含模态、DA状态、时间戳，便于追踪。

#### 4.15.2 断点续训一致性修复

在中期实验中，曾出现“恢复训练后曲线断崖”问题。复盘后定位为恢复流程未完整恢复 scheduler 状态。修复后采用“model + optimizer + scheduler”三状态强制恢复，曲线连续性显著改善。该经验在论文中属于关键工程贡献：它直接影响结论可比性。

#### 4.15.3 坏媒体样本治理

在大规模混合数据训练中，损坏视频/音频会导致数据加载阻塞、训练耗时异常波动。本文通过健康检查脚本在训练前过滤异常媒体，必要时迁移到坏样本目录，减少训练不确定性并提升整体吞吐效率。

### 4.16 模型与文献的完整映射（论文可直接引用）

| 模型/技术 | 文献来源 | 研究中的实现模块 |
|---|---|---|
| 情绪转变感知融合 | CFN-ESA | 情绪转变感知融合子模块 |
| 功能最大相关 | MFMC | 跨模态相关性约束子模块 |
| 领导-跟随融合 | Leader-Follower | 非对称引导融合子模块 |
| 两阶段图注意力融合 | GA2MIF | 两阶段关系融合子模块 |
| 文本编码 | BERT | 文本语义编码子模块 |
| 语音编码 | Wav2Vec2 | 语音上下文编码子模块 |
| 视频编码 | ResNet50 | 视觉特征编码子模块 |
| 跨模态注意力基础范式 | Transformer/MulT | 标准跨模态注意力融合子模块 |

该映射表用于说明“文献思想—方法模块—实验变量”的一致性，是学位论文方法可追溯的重要依据。

### 4.17 已完成工作全景总结（面向学位论文）

截至当前阶段，已形成以下可被论文明确陈述的工作闭环：

1. 完成统一主干模型的工程实现与配置化实验框架；
2. 完成 T/A/V/AT/VT 消融实验，并形成初步规律；
3. 完成 AT 的 noDA/DA 对照，验证 DA 并非必然增益；
4. 完成 CREMA 微调的 from-pretrain 与 from-scratch 对照；
5. 构建并验证数据健康检查与异常样本治理流程；
6. 修复断点续训一致性问题并沉淀复现规范；
7. 启动 AVT 主线实验并建立后续任务清单；
8. 形成文献-模型-代码-实验四位一体证据链。

### 4.18 面向终稿提交的补充建议（执行清单）

为使本文档达到最终学位论文提交标准，建议按以下顺序补完：

1. AVT 三条主实验（noDA、DA、emotion_shift）结果回填；
2. 在关键对照组上增加重复实验，报告均值与标准差；
3. 增加错误案例分析（失败样本可解释性）；
4. 增加训练效率统计（健康检查前后耗时对比）；
5. 增加章节间交叉引用（图表、公式、文献编号）。

### 4.19 提交版图表目录与编号方案

为避免终稿阶段图表编号混乱，建议统一采用“章内编号”规则。以下清单可直接作为论文图表目录草案：

#### 4.19.1 建议插入图（Figure）

- 图4-1：系统总体架构图（感知层-模型层-决策层）  
  - 来源：基于系统分层关系绘制；
- 图4-2：主模型前向数据流图（输入模态 -> 编码器 -> 融合 -> 任务头）  
  - 来源：基于模型前向流程与模态编码流程绘制；
- 图4-3：训练执行链路图（配置读取、训练循环、验证、日志、checkpoint）  
  - 来源：基于统一训练流程绘制；
- 图4-4：多数据集数据治理流程图（download/organize/merge/check/move_bad）  
  - 来源：基于数据治理流程绘制；
- 图4-5：断点续训一致性修复示意图（修复前后曲线对比）  
  - 来源：历史 logs 曲线；
- 图4-6：AVT 系列实验路线图（noDA/DA/emotion_shift）  
  - 来源：基于实验配置矩阵与执行计划绘制。

#### 4.19.2 建议插入表（Table）

- 表4-1：实验配置总览（配置文件、变量、目的）；
- 表4-2：模态消融结果（T/A/V/AT/VT/AVT）；
- 表4-3：DA 消融结果（AT/VT/AVT）；
- 表4-4：融合策略消融结果（standard / emotion_shift / leader_follower / two_stage）；
- 表4-5：微调范式对比（from-pretrain vs from-scratch）；
- 表4-6：工程稳定性指标（训练中断率、恢复成功率、坏样本占比、训练吞吐变化）。

#### 4.19.3 正文引用模板（可直接粘贴）

1. “如图4-2所示，模型在统一骨架下通过可插拔融合模块实现策略切换，确保了消融实验的单变量可控性。”  
2. “由表4-3可见，DA 在 AT 组合上未表现出稳定增益，提示其收益对模态组合与训练配方高度敏感。”  
3. “如图4-5所示，在恢复 scheduler 状态后，续训曲线不连续问题得到明显缓解。”

### 4.20 提交版结果叙述模板（按结果自动替换）

以下模板用于最终结果回填，避免临近答辩时出现文本逻辑不一致。

#### 4.20.1 主结果总述模板

“在统一训练配置下，本文比较了 {模态组合集合} 在 {数据集设置} 上的性能。结果显示，{最佳组合} 在 {指标1}/{指标2} 上取得最优，其中 {具体数值}。与基线 {基线组合} 相比，提升幅度为 {提升值}。该结果表明 {结论}。”

#### 4.20.2 模态消融模板

“模态消融结果表明，文本模态在跨数据集条件下表现最稳定（{指标}={数值}）。视频模态单独使用时性能较弱，但在与文本结合后出现互补增益（{指标变化}）。这说明视觉信息在当前数据质量条件下更适合作为辅助模态而非主导模态。”

#### 4.20.3 DA 对照模板

“在 {模态组合} 上，DA 组与 noDA 组相比，Best 指标变化为 {ΔBest}，Last 指标变化为 {ΔLast}。若 Best 与 Last 呈同向提升，可判定 DA 在该设置下有效；若仅出现短暂峰值提升而终点退化，则判定为不稳定收益。”

#### 4.20.4 融合策略模板

“在固定 {模态组合} 与 {训练配方} 条件下，`emotion_shift` 相较 `standard` 的 {指标} 变化为 {Δ}。若多次重复实验均表现为同向提升，说明情绪转变建模对该场景有效；否则建议保留 `standard` 作为工程默认策略。”

### 4.21 统计可靠性与显著性建议

为提高学位论文说服力，建议在关键对照实验（至少主结果、DA、融合策略三组）中加入统计可靠性报告：

1. 每组至少重复 3 次（建议 5 次）；
2. 报告 `mean ± std`；
3. 关键组间进行显著性检验（如 t-test）并报告 p 值；
4. 对结果波动较大的组，补充学习率曲线与样本质量解释。

可直接写入正文的句式：

“为验证结果稳定性，本文对关键实验重复 {N} 次，并报告均值与标准差。统计检验显示，{方法A} 相比 {方法B} 在 {指标} 上达到/未达到显著差异（p={p值}）。”

### 4.22 有效性威胁与缓解策略

本研究在终稿中建议单列“有效性威胁”小节，以提升学术严谨性。

1. **内部有效性威胁**：不同实验若存在隐含变量（如恢复策略不一致），会导致结论偏差。  
   - 缓解：统一训练入口、统一配置模板、统一日志格式。

2. **外部有效性威胁**：当前结论主要来自 CREMA-D、MELD、CMU-MOSEI 的组合，迁移到其他驾驶数据场景时可能存在分布差异。  
   - 缓解：增加跨数据集外部验证与多次重复统计。

3. **构念有效性威胁**：单一指标无法完整反映模型质量。  
   - 缓解：同时报告 Best 与 Last、分类与稳定性、精度与训练成本。

4. **结论有效性威胁**：小样本重复次数不足可能导致过度结论。  
   - 缓解：关键对照组补足重复实验并报告显著性。

### 4.23 最终答辩版章节联动说明

为保证答辩展示与论文正文一致，建议采用如下联动规则：

1. PPT“方法页”严格对应第3章（模块图 + 文献映射表）；
2. PPT“实验页”严格对应第4章（表4-1~表4-6 + 图4-1~图4-6）；
3. 答辩中所有结论句均可回溯到 `logs/*/metrics.csv` 与配置文件；
4. 对“进行阶段实验”采用明确表述：处于进行阶段、不下最终结论、预计补充时间点。

### 4.24 本章小结

本章系统给出了实验设计、执行过程与阶段结果，并从工程可复现性角度解释了训练稳定性与结论可信度。通过配置驱动实验矩阵、统一日志规范与异常治理流程，本文实现了“实验可追踪、结果可对照、结论可复核”的研究闭环。针对终稿提交，本章进一步提供了图表编号、结果回填与统计检验模板，可直接用于最终论文定稿。

## 第5章 总结与展望

本章对全文工作进行归纳，概括本文在模型方法与工程实践两方面的阶段性成果，并给出下一步研究方向与落地建议。

### 5.1 研究总结

本文构建了一个统一、可复现的多模态情绪识别框架，并围绕模态组合、域适应、融合策略和训练范式开展系统消融。实验表明，同一模型骨架下的配置驱动方法能够有效支撑科学对照与论文写作；文本模态稳定性强，多模态互补具备潜力，工程可靠性是确保结论可信的重要前提。

### 5.2 存在不足

1. 视频模态对数据质量敏感，单模态表现仍弱；
2. DA 与高级融合策略收益受配置与数据状态影响较大；
3. 关键实验重复次数仍有提升空间。

### 5.3 后续工作

1. 扩展 AVT 下更多融合策略对照（leader_follower / two_stage）；
2. 在统一清洗数据上进行多次重复统计均值与方差；
3. 探索更强视频主干或参数高效微调策略；
4. 拓展到更多下游单数据集迁移验证。

### 5.4 论文与工程产出清单

本文阶段性产出不仅包括算法对照结果，也包括可复现实验基础设施。可在结题或答辩材料中作为成果清单：

1. 统一主干模型与可配置融合框架；
2. 多数据集训练配置矩阵与标准化运行流程；
3. 断点续训一致性修复机制；
4. 数据健康检查与坏样本治理工具链；
5. 文献-模型-代码-实验四位一体映射体系；
6. 可直接复用的论文写作模板（结果叙述、图表引用、对照结论模板）。

### 5.5 本章小结

本文在统一主模型框架下完成了多模态、域适应、融合策略与训练范式的系统化对照，并通过工程闭环机制提升了实验结论的可解释性与可复现性。后续工作将围绕 AVT 主线补全、重复实验统计与跨场景外部验证展开，以形成更完整、更具推广价值的学位论文终稿。

---

## 参考文献

[1] Picard R W. *Affective Computing* [M]. Cambridge, MA: MIT Press, 1997.

[2] Devlin J, Chang M W, Lee K, et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding [C]//Proceedings of NAACL-HLT. 2019: 4171-4186.

[3] Mikolov T, Sutskever I, Chen K, et al. Distributed Representations of Words and Phrases and their Compositionality [C]//Advances in Neural Information Processing Systems (NeurIPS). 2013.

[4] He K, Zhang X, Ren S, et al. Deep Residual Learning for Image Recognition [C]//Proceedings of CVPR. 2016: 770-778.

[5] Baevski A, Zhou Y, Mohamed A, et al. wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations [C]//Advances in Neural Information Processing Systems (NeurIPS). 2020.

[6] Vaswani A, Shazeer N, Parmar N, et al. Attention Is All You Need [C]//Advances in Neural Information Processing Systems (NeurIPS). 2017.

[7] Tsai Y H H, Bai S, Liang P P, et al. Multimodal Transformer for Unaligned Multimodal Language Sequences [C]//Proceedings of ACL. 2019: 6558-6569.

[8] Li J, Wang X, Liu Y, et al. CFN-ESA: A Cross-Modal Fusion Network with Emotion-Shift Awareness for Dialogue Emotion Recognition [J]. arXiv preprint arXiv:2307.15432, 2023.

[9] Li J, Wang X, Lv G, et al. GA2MIF: Graph and Attention Based Two-Stage Multi-Source Information Fusion for Conversational Emotion Detection [J]. arXiv preprint arXiv:2207.11900, 2022.

[10] Zhang S, Ding Y, Wei Z, et al. Continuous Emotion Recognition with Audio-Visual Leader-Follower Attentive Fusion [C]//Proceedings of ICCV. 2021: 3557-3566.

[11] Zheng D, Zhang T, Zheng W, et al. Multimodal Functional Maximum Correlation for Emotion Recognition [J]. arXiv preprint arXiv:2512.23076, 2025.

[12] Zadeh A, Liang P P, Poria S, et al. Multimodal Language Analysis in the Wild: CMU-MOSEI Dataset and Interpretable Dynamic Fusion Graph [C]//Proceedings of ACL. 2018: 2236-2246.

[13] Busso C, Bulut M, Lee C C, et al. IEMOCAP: Interactive Emotional Dyadic Motion Capture Database [J]. Language Resources and Evaluation, 2008, 42(4): 335-359.

[14] Poria S, Hazarika D, Majumder N, et al. MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations [C]//Proceedings of ACL. 2019: 527-536.

[15] Cao H, Cooper D G, Keutmann M K, et al. CREMA-D: Crowd-Sourced Emotional Multimodal Actors Dataset [J]. IEEE Transactions on Affective Computing, 2014, 5(4): 377-390.

[16] Tang L, Li Y, Yuan J, et al. CPSOR-GCN: A Vehicle Trajectory Prediction Method Powered by Emotion and Cognitive Theory [J]. arXiv preprint arXiv:2311.08086, 2023.

[17] Xing Y, Lv C, Cao D, et al. A Unified Multi-Scale and Multi-Task Learning Framework for Driver Behaviors Reasoning [J]. IEEE Transactions on Intelligent Transportation Systems, 2020, 21(11): 4642-4654.

[18] Avetisyan L, Yang X J, Zhou F. Towards Context-Aware Modeling of Situation Awareness in Conditionally Automated Driving [J]. arXiv preprint arXiv:2405.07088, 2024.

[19] Hazarika D, Zimmermann R, Poria S. MISA: Modality-Invariant and -Specific Representations for Multimodal Sentiment Analysis [C]//Proceedings of ACM Multimedia. 2020: 1122-1131.

[20] Mittal T, Bhattacharya U, Chandra R, et al. M3ER: Multiplicative Multimodal Emotion Recognition Using Facial, Textual, and Speech Cues [C]//AAAI Workshop / arXiv:1911.05659. 2019.

### 文中引用写法建议

1. 单篇引用示例：  
   “BERT 通过双向上下文建模显著提升了文本语义表示能力[2]。”

2. 多篇并列引用示例：  
   “跨模态融合已从简单拼接发展到注意力与图建模范式[7,9,10,19]。”

3. 方法来源声明示例：  
   “本文 `emotion_shift` 融合策略主要参考 CFN-ESA 的情绪转变感知思想[8]；`leader_follower` 策略参考领导-跟随机制[10]；`two_stage` 策略参考两阶段图注意力融合方法[9]；相关性约束参考 MFMC 思路[11]。”

4. 数据集引用示例：  
   “本研究主要使用 CREMA-D、MELD 与 CMU-MOSEI 构建跨数据集训练与评估流程[12,14,15]。”

> 注：若学校或学院要求 GB/T 7714-2015（顺序编码制）严格格式，可在定稿阶段将以上条目导入 Zotero/NoteExpress 自动排版为最终版。

### 终版排版建议清单（提交前逐项核对）

1. **封面与摘要一致性**：中文题目、英文题目、关键词中英文数量与顺序一致；  
2. **编号一致性**：图、表、公式按“章内编号”统一（如图4-1、表4-1、式(3-1)）；  
3. **引用一致性**：正文中的 `[n]` 均能在参考文献中找到对应条目，且无孤立条目；  
4. **术语一致性**：统一使用 `from-pretrain/from-scratch`、`noDA/DA`、`Best/Last`；  
5. **结果一致性**：正文叙述、表格数值、图中标注三者一致；  
6. **阶段性声明**：未完成实验统一使用“处于进行阶段/待回填/不下最终结论”；  
7. **附录一致性**：命令、配置名、日志目录与正文保持一致；  
8. **提交格式**：按学校模板统一页边距、字体、行距、标题层级与参考文献样式。

## 附录C：论文目录草案（到三级标题）

> 说明：以下目录草案与当前文档内容一一对应，可直接迁移至学校论文模板，并在排版工具中自动生成页码。

### 摘要
### Abstract

### 第1章 绪论
#### 1.1 研究背景与意义
#### 1.2 研究问题
#### 1.3 研究内容与贡献
#### 1.4 论文结构
#### 1.5 本章小结

### 第2章 相关技术与研究现状
#### 2.1 多模态情绪识别
#### 2.2 融合策略
#### 2.3 域适应与类别不平衡
#### 2.4 文本表示方法：Word2Vec 与 BERT
#### 2.5 研究不足与本文切入点
#### 2.6 本章小结

### 第3章 方法设计
#### 3.1 整体框架
#### 3.2 特征提取器
#### 3.3 融合模块
#### 3.4 损失函数与优化目标
#### 3.5 训练与恢复规范
#### 3.6 文献到实现的映射关系
#### 3.7 数据流与前向计算过程（实现视角）
#### 3.8 训练策略扩展（策略工厂）
#### 3.9 关键算法复杂度与工程代价分析
#### 3.10 数据工程与标签统一策略
#### 3.11 本章小结

### 第4章 实验设置与结果分析
#### 4.1 实验环境与实现细节
#### 4.2 数据集与统一预处理
#### 4.3 实验变量与对照原则
#### 4.4 已完成实验概览（基于当前工程进度）
#### 4.5 待完成实验（按论文主线）
#### 4.6 实验顺序（严格执行版）
#### 4.7 结果记录模板（已自动回填已完成实验）
#### 4.8 图表编排建议（含正文引用模板）
#### 4.9 结果分析写作模板（可直接替换数字）
#### 4.10 准提交版结论段（已按当前结果自动生成，可直接粘贴正文）
#### 4.11 项目代码结构与系统分层设计
#### 4.12 模型层详细实现（与论文方法一一对应）
#### 4.13 脚本层与实验执行链路
#### 4.14 配置驱动实验矩阵（全量映射）
#### 4.15 训练过程、异常处理与复现实践
#### 4.16 模型与文献的完整映射（论文可直接引用）
#### 4.17 已完成工作全景总结（面向学位论文）
#### 4.18 面向终稿提交的补充建议（执行清单）
#### 4.19 提交版图表目录与编号方案
#### 4.20 提交版结果叙述模板（按结果自动替换）
#### 4.21 统计可靠性与显著性建议
#### 4.22 有效性威胁与缓解策略
#### 4.23 最终答辩版章节联动说明
#### 4.24 本章小结

### 第5章 总结与展望
#### 5.1 研究总结
#### 5.2 存在不足
#### 5.3 后续工作
#### 5.4 论文与工程产出清单
#### 5.5 本章小结

### 参考文献
### 附录A：实验执行总清单（论文前核对）
### 附录B：核心命令模板
### 附录C：论文目录草案（到三级标题）
### 附录D：图表目录草案
### 附录E：缩略语表草案

## 附录D：图表目录草案

### D.1 插图目录草案（Figure List）

- 图3-1 多模态情绪识别整体方法框架图  
- 图3-2 主模型前向数据流图  
- 图3-3 多模态融合策略切换示意图（standard/emotion_shift/leader_follower/two_stage）  
- 图3-4 数据工程流程图（download-organize-merge-check-clean）  
- 图4-1 训练损失曲线对比图（AT/VT/AVT）  
- 图4-2 验证集 F1 曲线对比图  
- 图4-3 断点续训修复前后曲线对比图  
- 图4-4 坏样本治理前后训练效率对比图  
- 图4-5 AVT 系列实验执行路线图  
- 图4-6 关键实验结果可视化对比图（可选柱状图/折线图）

### D.2 表格目录草案（Table List）

- 表3-1 文献-模型-代码映射表  
- 表3-2 关键模块时间复杂度与工程代价对比表  
- 表4-1 主结果总览（终稿回填模板）  
- 表4-2 模态消融结果表  
- 表4-3 DA 消融结果表  
- 表4-4 融合策略消融结果表  
- 表4-5 微调范式对比表（from-pretrain vs from-scratch）  
- 表4-6 工程复现性与稳定性指标表（恢复成功率/坏样本比例/吞吐变化）  
- 表5-1 论文与工程产出清单表（可选）

### D.3 公式目录草案（Equation List，可选）

- 式(3-1) 总损失函数  
- 式(3-2) 域对抗损失项  
- 式(3-3) 相关性约束项  
- 式(3-4) 指标统计表达（mean ± std）

## 附录E：缩略语表草案

| 缩略语 | 英文全称 | 中文释义 |
|---|---|---|
| MER | Multimodal Emotion Recognition | 多模态情绪识别 |
| HMI | Human-Machine Interaction | 人机交互 |
| DA | Domain Adaptation | 域适应 |
| DMS | Driver Monitoring System | 驾驶员监测系统 |
| GRL | Gradient Reversal Layer | 梯度反转层 |
| FMC | Functional Maximum Correlation | 功能最大相关 |
| MFMC | Multimodal Functional Maximum Correlation | 多模态功能最大相关 |
| AT | Audio-Text | 音频-文本模态组合 |
| VT | Video-Text | 视频-文本模态组合 |
| AVT | Audio-Video-Text | 音频-视频-文本模态组合 |
| CE | Cross Entropy | 交叉熵 |
| Focal Loss | Focal Loss | 焦点损失 |
| CB Loss | Class-Balanced Loss | 类别平衡损失 |
| ASR | Automatic Speech Recognition | 自动语音识别 |
| NLP | Natural Language Processing | 自然语言处理 |
| CNN | Convolutional Neural Network | 卷积神经网络 |
| BiLSTM | Bidirectional Long Short-Term Memory | 双向长短期记忆网络 |
| SOTA | State-of-the-Art | 当前最优水平 |
| SDK | Software Development Kit | 软件开发工具包 |

> 建议：在正文首次出现缩略语时采用“中文全称（英文全称，缩略语）”写法，后续统一使用缩略语。

---

## 附录A：实验执行总清单（论文前核对）

- [ ] 模态主线：T/A/V/AT/VT/AVT 全部完成
- [ ] DA主线：AT/VT/AVT 的 noDA vs DA 全部完成
- [ ] 融合主线：AVT standard vs emotion_shift 完成
- [ ] 训练范式：from-pretrain vs from-scratch 至少两组对照
- [ ] 每组实验具备：配置文件、run名、metrics.csv、TensorBoard截图
- [ ] 异常问题具备：现象、原因、修复、影响记录

## 附录B：核心命令模板

```bash
# AVT noDA
python3 scripts/train.py --config config/config_AVT_noDA.yaml --mode pretrain

# AVT DA
python3 scripts/train.py --config config/config_AVT_DA.yaml --mode pretrain

# AVT noDA + emotion_shift
python3 scripts/train.py --config config/config_AVT_noDA_emotion_shift.yaml --mode pretrain
```

```bash
# 续训（示例）
python3 scripts/train.py --config config/config_AVT_noDA.yaml --mode pretrain --resume checkpoints/checkpoint_pretrain_epoch_XX.pth
```

```bash
# TensorBoard
tensorboard --logdir logs --host 0.0.0.0 --port 6007
```

---

> 使用说明：本文档是“可直接写入论文正文”的完整草稿。你后续仅需把表格中的数值替换为最终实验结果，并结合图表补充分析细节，即可形成可提交版本。
