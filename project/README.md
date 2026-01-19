# 多模态驾驶员情绪分析项目

## 项目概述

本项目是一个基于情感计算的人机交互智能体系统，聚焦于智能驾驶领域，通过对驾驶员的多模态数据（视频、语音、文本、生理信号）进行综合分析，实现准确的驾驶员情绪识别和监测。项目集成了多篇顶会论文的优秀方法，旨在为智能驾驶系统提供实时、准确的驾驶员情绪分析能力，从而提升驾驶安全性和用户体验。

### 项目目的和意义

1. **提升驾驶安全性**：通过实时监测驾驶员情绪状态，及时发现疲劳、焦虑、愤怒等负面情绪，提前预警潜在危险
2. **优化人机交互**：根据驾驶员情绪状态，智能调整车载系统交互方式，提供个性化服务
3. **推动智能驾驶发展**：将情感计算与智能驾驶结合，为自动驾驶和辅助驾驶系统提供重要的感知能力
4. **学术研究价值**：整合多模态情感分析领域的最新研究成果，为相关研究提供可复现的代码实现

---

## 借鉴的论文与方法

本项目深度借鉴了多篇CCF-A类会议和顶级期刊论文的优秀方法，以下是详细的论文列表和借鉴内容：

### 1. CFN-ESA: 情感转变感知机制 ⭐⭐⭐⭐⭐

**论文信息**：
- **标题**：CFN-ESA: A Cross-Modal Fusion Network with Emotion-Shift Awareness for Dialogue Emotion Recognition
- **作者**：Jiang Li, Xiaoping Wang, Yingjian Liu, Zhigang Zeng
- **发表时间**：2023年7月
- **论文链接**：https://arxiv.org/abs/2307.15432
- **源码地址**：https://github.com/jianglil/Cross-Modal-Fusion-Network

**借鉴的方法**：
- **情感转变感知模块（EmotionShiftAwareness）**：捕捉时序中的情感变化模式，使用LSTM检测情感转变，适用于驾驶员情绪的动态监测
- **跨模态融合网络（EmotionShiftFusion）**：将文本模态作为主要情感信息来源，其他模态作为次要来源，通过加权融合和跨模态注意力实现多模态特征融合
- **情感转变权重计算**：通过相邻时间步的情感差异计算转变强度，帮助识别情绪突变

**实现位置**：`models/emotion_shift.py`

**应用场景**：适用于需要捕捉驾驶员情绪动态变化的场景，如长时间驾驶监测、情绪状态演变分析

---

### 2. MFMC: 多模态功能最大相关 ⭐⭐⭐⭐⭐

**论文信息**：
- **标题**：Multimodal Functional Maximum Correlation for Emotion Recognition
- **作者**：Deyang Zheng, Tianyi Zhang, Wenming Zheng, Shujian Yu
- **发表时间**：2025年12月（预印本）
- **论文链接**：https://arxiv.org/abs/2512.23076
- **源码地址**：https://github.com/DY9910/MFMC

**借鉴的方法**：
- **功能最大相关（FunctionalMaximumCorrelation）**：通过最大化不同模态之间的相关性来学习更好的多模态表示
- **多模态相关性损失（MultimodalCorrelationLoss）**：用于预训练阶段，通过最大化多模态依赖性提高情感识别准确性
- **自监督学习框架**：可用于预训练阶段，无需大量标注数据即可学习多模态特征表示

**实现位置**：`models/functional_correlation.py`

**应用场景**：预训练阶段，通过最大化多模态相关性学习通用特征表示，提高模型泛化能力

---

### 3. Continuous Emotion Recognition: 领导-跟随注意力 ⭐⭐⭐⭐

**论文信息**：
- **标题**：Continuous Emotion Recognition with Audio-visual Leader-follower Attentive Fusion
- **作者**：Su Zhang, Yi Ding, Ziquan Wei, Cuntai Guan
- **发表时间**：2021年7月
- **会议**：ICCV 2021（CCF-A类）
- **论文链接**：https://arxiv.org/abs/2107.01175
- **源码地址**：https://github.com/sucv/ABAW2

**借鉴的方法**：
- **领导-跟随注意力机制（LeaderFollowerAttention）**：一个模态作为"领导者"引导另一个模态的特征学习，允许模态之间相互引导
- **双向领导-跟随（BidirectionalLeaderFollower）**：允许两个模态相互引导，提高融合效果
- **多模态领导-跟随融合（MultimodalLeaderFollowerFusion）**：扩展到多个模态，在智能驾驶场景中设置文本或生理信号为领导者

**实现位置**：`models/leader_follower_attention.py`

**应用场景**：适用于连续情感识别任务，可以设置文本或生理信号为领导者，引导其他模态的特征学习

---

### 4. GA2MIF: 两阶段融合策略 ⭐⭐⭐⭐

**论文信息**：
- **标题**：GA2MIF: Graph and Attention Based Two-Stage Multi-Source Information Fusion for Conversational Emotion Detection
- **作者**：Jiang Li, Xiaoping Wang, Guoqing Lv, Zhigang Zeng
- **发表时间**：2022年7月
- **论文链接**：https://arxiv.org/abs/2207.11900

**借鉴的方法**：
- **两阶段融合（TwoStageFusion）**：
  - **第一阶段**：使用图注意力网络（GAT）进行上下文建模，捕捉模态之间的图结构关系
  - **第二阶段**：使用跨模态注意力进行多模态融合，更好地捕捉模态内和模态间的依赖关系
- **图注意力层（GraphAttentionLayer）**：建模模态之间的图结构关系，使用自注意力模拟图注意力机制

**实现位置**：`models/two_stage_fusion.py`

**应用场景**：适用于需要同时考虑模态内和模态间依赖关系的复杂多模态融合场景

---

### 5. 其他参考论文

#### CPSOR-GCN: 情绪对驾驶行为的影响 ⭐⭐⭐⭐⭐
- **论文**：CPSOR-GCN: A Vehicle Trajectory Prediction Method Powered by Emotion and Cognitive Theory
- **核心思想**：结合图卷积网络（GCN）和动态贝叶斯网络（DBN）量化情绪对驾驶行为的影响
- **借鉴点**：为驾驶员情绪分析在智能驾驶中的应用提供重要参考，GCN可用于建模驾驶员-车辆-环境的交互关系

#### Unified Multi-scale Framework: 多任务学习 ⭐⭐⭐⭐⭐
- **论文**：A Unified Multi-scale and Multi-task Learning Framework for Driver Behaviors Reasoning
- **核心思想**：统一的多尺度、多任务学习框架，同时识别驾驶员的身体姿势、意图和情绪
- **借鉴点**：多任务学习框架可扩展应用到多模态情绪分析，多尺度设计有助于捕捉不同时间尺度的情绪变化

---

## 技术框架

### 整体架构

```
输入层（多模态数据）
    ├── 视频模态：驾驶员面部视频 → ResNet-50 → 视频特征
    ├── 语音模态：驾驶员语音/音频 → Wav2Vec2 → 语音特征
    ├── 生理信号：EEG、ECG、GSR、眼动数据 → 1D-CNN/LSTM → 生理特征
    └── 文本模态：语音转文本或交互文本 → BERT → 文本特征

特征提取层（预训练模型）
    ├── 视频特征提取器：ResNet-50 → 面部表情特征
    ├── 语音特征提取器：Wav2Vec2 → 语音情感特征
    ├── 生理信号特征提取器：1D-CNN/LSTM → 生理特征
    └── 文本特征提取器：BERT → 文本语义特征

多模态融合层（核心模块，支持多种策略）
    ├── 标准融合（MultimodalFusion）：基础的多模态融合
    ├── 情感转变感知融合（EmotionShiftFusion）：CFN-ESA方法
    ├── 领导-跟随注意力融合（MultimodalLeaderFollowerFusion）：Continuous Emotion Recognition方法
    └── 两阶段融合（TwoStageFusion）：GA2MIF方法

输出层
    ├── 情绪分类器：离散情绪类别（快乐、悲伤、愤怒、恐惧、中性、焦虑等）
    ├── 情绪强度回归器：连续情绪维度（效价、唤醒度）
    └── 情绪趋势预测器：时序情绪变化预测
```

### 融合策略选择

项目支持四种融合策略，可通过配置文件选择：

1. **standard**：标准的多模态融合（基础方法）
2. **emotion_shift**：情感转变感知融合（推荐，CFN-ESA方法）
3. **leader_follower**：领导-跟随注意力融合（Continuous Emotion Recognition方法）
4. **two_stage**：两阶段融合（GA2MIF方法）

### 训练策略

1. **预训练阶段**：
   - 使用通用多模态情感数据集（MAFW、AffectNet、IEMOCAP）
   - 可选：使用功能最大相关损失（MFMC）进行自监督预训练
   - 学习多模态特征提取和基础融合能力

2. **微调阶段**：
   - 使用驾驶员专用数据集（MPDB、DEFE）
   - 适配驾驶场景的特殊性
   - 优化模型在真实驾驶环境下的表现

---

## 项目结构

```
project/
├── models/                          # 模型定义
│   ├── __init__.py                  # 模块导出
│   ├── feature_extractors.py        # 多模态特征提取器
│   │   ├── VideoFeatureExtractor    # 视频特征提取（ResNet-50）
│   │   ├── AudioFeatureExtractor    # 语音特征提取（Wav2Vec2）
│   │   ├── PhysiologicalFeatureExtractor  # 生理信号特征提取（1D-CNN/LSTM）
│   │   └── TextFeatureExtractor     # 文本特征提取（BERT）
│   ├── attention_modules.py         # 基础注意力融合模块
│   │   ├── MultiHeadSelfAttention   # 多头自注意力
│   │   ├── CrossModalAttention      # 跨模态注意力
│   │   ├── TemporalAttention        # 时序注意力
│   │   └── MultimodalFusion         # 标准多模态融合
│   ├── emotion_shift.py             # 情感转变感知模块（CFN-ESA）
│   │   ├── EmotionShiftAwareness    # 情感转变感知
│   │   └── EmotionShiftFusion       # 情感转变融合
│   ├── leader_follower_attention.py # 领导-跟随注意力（Continuous Emotion Recognition）
│   │   ├── LeaderFollowerAttention # 领导-跟随注意力
│   │   ├── BidirectionalLeaderFollower  # 双向领导-跟随
│   │   └── MultimodalLeaderFollowerFusion  # 多模态领导-跟随融合
│   ├── two_stage_fusion.py         # 两阶段融合（GA2MIF）
│   │   ├── GraphAttentionLayer     # 图注意力层
│   │   └── TwoStageFusion          # 两阶段融合
│   ├── functional_correlation.py   # 功能最大相关（MFMC）
│   │   ├── FunctionalMaximumCorrelation  # 功能最大相关
│   │   └── MultimodalCorrelationLoss     # 多模态相关性损失
│   └── multimodal_model.py         # 完整的多模态情绪分析模型
│       └── MultimodalEmotionModel   # 主模型类
├── data/                            # 数据处理和加载
│   ├── __init__.py
│   ├── train.csv                    # 训练集索引
│   ├── val.csv                      # 验证集索引
│   ├── test.csv                     # 测试集索引
│   ├── video/                       # 视频数据
│   ├── audio/                       # 音频数据
│   ├── physiological/               # 生理信号数据
│   ├── text/                        # 文本数据
│   ├── labels/                      # 标签数据
│   └── dataset.py                   # 多模态数据集加载器
│       └── MultimodalDataset        # 数据集类
├── config/                          # 配置文件
│   └── config.yaml                 # 模型和训练配置
├── utils/                           # 工具函数
│   ├── __init__.py
│   └── helpers.py                   # 辅助函数（配置加载、设备管理等）
├── scripts/                         # 训练和推理脚本
│   ├── train.py                    # 训练脚本
│   └── inference.py                 # 推理脚本
├── requirements.txt                 # Python依赖包
├── .gitignore                       # Git忽略文件
└── README.md                        # 项目说明文档
```

---

## 使用方法

### 1. 环境配置

```bash
# 安装依赖
pip install -r requirements.txt
```

### 2. 配置文件

编辑 `config/config.yaml` 选择融合策略：

```yaml
model:
  attention:
    fusion_strategy: "emotion_shift"  # 可选: standard, emotion_shift, leader_follower, two_stage
    use_fmc_loss: true  # 预训练阶段建议开启功能最大相关损失
```

### 3. 训练模型

```bash
# 预训练阶段
python scripts/train.py --config config/config.yaml --mode pretrain

# 微调阶段
python scripts/train.py --config config/config.yaml --mode finetune
```

### 4. 推理

```bash
python scripts/inference.py \
    --model_path checkpoints/best_model.pth \
    --input_path data/test/ \
    --output_path results/
```

---

## 数据集

### 预训练阶段
- **MAFW**：多模态情感数据库（面部+语音）
- **AffectNet**：大规模面部表情数据库
- **IEMOCAP**：交互式情感对话数据库

### 微调阶段
- **MPDB**：多模态生理驾驶员行为数据库（生理信号）
- **DEFE**：驾驶员情感面部表情数据集（如可获得）

详细的数据集获取方法和使用指南请参考：
- `../dataset_application_guide.md`：数据集应用指南
- `../links.txt`：数据集链接和获取方式

---

## 技术栈

### 深度学习框架
- **PyTorch**：主要深度学习框架

### 预训练模型
- **视频**：ResNet-50（ImageNet预训练）
- **语音**：Wav2Vec2（Hugging Face）
- **文本**：BERT（Hugging Face）
- **生理信号**：自定义1D-CNN/LSTM架构

### 工具库
- **数据处理**：NumPy, Pandas, OpenCV
- **音频处理**：Librosa, torchaudio
- **生理信号处理**：MNE-Python, SciPy
- **NLP**：Transformers (Hugging Face)
- **可视化**：Matplotlib, Seaborn, TensorBoard

---

## 评估指标

- **分类任务**：准确率（Accuracy）、精确率（Precision）、召回率（Recall）、F1分数
- **回归任务**：平均绝对误差（MAE）、均方根误差（RMSE）、相关系数（Correlation）
- **多模态融合效果**：各模态贡献度分析、消融实验

---

## 参考文档

- **研究指导方案**：`../research_guide.md`
- **数据集应用指南**：`../dataset_application_guide.md`
- **论文指南**：`../article_guide.md`
- **数据集链接**：`../links.txt`

---

## 贡献与引用

### 引用本项目

如果您在研究中使用了本项目，请引用：

```bibtex
@software{multimodal_driver_emotion,
  title={多模态驾驶员情绪分析项目},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo}
}
```

### 引用相关论文

本项目借鉴了以下论文，请在使用相关方法时引用原始论文：

1. **CFN-ESA**:
```bibtex
@article{li2023cfn,
  title={CFN-ESA: A Cross-Modal Fusion Network with Emotion-Shift Awareness for Dialogue Emotion Recognition},
  author={Li, Jiang and Wang, Xiaoping and Liu, Yingjian and Zeng, Zhigang},
  journal={arXiv preprint arXiv:2307.15432},
  year={2023}
}
```

2. **MFMC**:
```bibtex
@article{zheng2025mfmc,
  title={Multimodal Functional Maximum Correlation for Emotion Recognition},
  author={Zheng, Deyang and Zhang, Tianyi and Zheng, Wenming and Yu, Shujian},
  journal={arXiv preprint arXiv:2512.23076},
  year={2025}
}
```

3. **Continuous Emotion Recognition**:
```bibtex
@inproceedings{zhang2021continuous,
  title={Continuous Emotion Recognition with Audio-visual Leader-follower Attentive Fusion},
  author={Zhang, Su and Ding, Yi and Wei, Ziquan and Guan, Cuntai},
  booktitle={ICCV},
  year={2021}
}
```

4. **GA2MIF**:
```bibtex
@article{li2022ga2mif,
  title={GA2MIF: Graph and Attention Based Two-Stage Multi-Source Information Fusion for Conversational Emotion Detection},
  author={Li, Jiang and Wang, Xiaoping and Lv, Guoqing and Zeng, Zhigang},
  journal={arXiv preprint arXiv:2207.11900},
  year={2022}
}
```

---

## 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。

---

## 联系方式

如有问题或建议，请通过以下方式联系：
- **Issues**：在GitHub上提交Issue
- **Email**：your-email@example.com

---

**最后更新**：2024年
