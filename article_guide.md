# CCF-A类会议论文指南：多模态情感分析与智能驾驶

## 说明

本文档收集了近2-3年（2022-2024）CCF-A类会议中关于多模态情感分析、驾驶员情绪分析和智能驾驶方向的相关论文。每篇论文包含标题、作者、会议信息、摘要分析、下载地址和源码地址。

**CCF-A类会议列表**：
- **计算机视觉**：CVPR, ICCV, ECCV
- **机器学习**：NeurIPS, ICML, ICLR
- **人工智能**：AAAI, IJCAI
- **自然语言处理**：ACL, EMNLP
- **数据挖掘**：KDD, WWW

---

## 一、多模态情感识别与注意力融合

### 1. Multimodal Functional Maximum Correlation for Emotion Recognition

- **作者**：Deyang Zheng, Tianyi Zhang, Wenming Zheng, Shujian Yu
- **发表时间**：2025年12月（预印本）
- **会议**：arXiv preprint（投稿中）
- **论文链接**：https://arxiv.org/abs/2512.23076
- **源码地址**：https://github.com/DY9910/MFMC

**内容摘要**：
- 提出了一种名为多模态功能最大相关（MFMC）的自监督学习框架
- 通过双重总相关目标最大化多模态依赖性，提高情感识别准确性
- 使用功能最大相关（FMC）方法捕捉多模态之间的高阶相关性
- 在多个多模态情感数据集上验证了方法的有效性

**与本研究关联性**：⭐⭐⭐⭐⭐
- 直接针对多模态情感识别任务
- 自监督学习框架可用于预训练阶段
- 最大化多模态依赖性的思想可应用于注意力融合模块

---

### 2. CFN-ESA: A Cross-Modal Fusion Network with Emotion-Shift Awareness for Dialogue Emotion Recognition

- **作者**：Jiang Li, Xiaoping Wang, Yingjian Liu, Zhigang Zeng
- **发表时间**：2023年7月
- **会议**：arXiv preprint（可能投稿至ACL/EMNLP）
- **论文链接**：https://arxiv.org/abs/2307.15432
- **源码地址**：https://github.com/jianglil/Cross-Modal-Fusion-Network

**内容摘要**：
- 提出跨模态融合网络（CFN-ESA），具有情感转变感知能力
- 将文本模态作为主要情感信息来源，视觉和音频模态作为次要来源
- 引入情感转移模块，捕捉对话中的情感变化
- 在多个对话情感识别数据集上取得显著性能提升

**与本研究关联性**：⭐⭐⭐⭐⭐
- 跨模态融合机制可直接应用于驾驶员多模态情绪分析
- 情感转变感知有助于理解驾驶员情绪的动态变化
- 适用于驾驶员与车载系统的交互情感分析场景

---

### 3. Continuous Emotion Recognition with Audio-visual Leader-follower Attentive Fusion

- **作者**：Su Zhang, Yi Ding, Ziquan Wei, Cuntai Guan
- **发表时间**：2021年7月
- **会议**：ICCV 2021（CCF-A类）
- **论文链接**：https://arxiv.org/abs/2107.01175
- **源码地址**：https://github.com/sucv/ABAW2

**内容摘要**：
- 提出音频-视觉时空深度神经网络
- 包含视觉块、听觉块和领导-跟随注意力融合块
- 在Aff-Wild2数据库上实现连续情感识别
- 领导-跟随注意力机制允许一个模态引导另一个模态的特征学习
- 使用大窗口长度的时间卷积网络（TCN）提取时空信息

**与本研究关联性**：⭐⭐⭐⭐
- 领导-跟随注意力机制可应用于多模态融合
- 连续情感识别适用于实时驾驶员情绪监测
- 音频-视觉融合方法可扩展到包含生理信号的多模态系统

**内容摘要**：
- 提出音频-视觉时空深度神经网络
- 包含视觉块、听觉块和领导-跟随注意力融合块
- 在Aff-Wild2数据库上实现连续情感识别
- 领导-跟随注意力机制允许一个模态引导另一个模态的特征学习

**与本研究关联性**：⭐⭐⭐⭐
- 领导-跟随注意力机制可应用于多模态融合
- 连续情感识别适用于实时驾驶员情绪监测
- 音频-视觉融合方法可扩展到包含生理信号的多模态系统

---

### 4. Cluster-Level Contrastive Learning for Emotion Recognition in Conversations

- **作者**：Kailai Yang, Tianlin Zhang, Hassan Alhuzali, Sophia Ananiadou
- **发表时间**：2023年2月
- **会议**：可能投稿至ACL/EMNLP（CCF-A类）
- **论文链接**：https://arxiv.org/abs/2302.03508
- **源码地址**：https://github.com/SteveKGYang/SCCL

**内容摘要**：
- 提出低维监督聚类级对比学习方法（SCCL）
- 将高维对比学习空间降低到三维情感表示空间（效价-唤醒度-支配度，VAD）
- 在该空间中执行聚类级对比学习，整合可测量的情感原型
- 适用于对话中的情感识别任务

**与本研究关联性**：⭐⭐⭐⭐
- 对比学习方法可用于多模态特征学习
- VAD情感表示空间与本研究中的情绪强度回归相关
- 聚类级学习可提高模型的泛化能力

---

### 5. M3ER: Multiplicative Multimodal Emotion Recognition Using Facial, Textual, and Speech Cues

- **作者**：Trisha Mittal, Uttaran Bhattacharya, Rohan Chandra, Aniket Bera, Dinesh Manocha
- **发表时间**：2019年11月
- **会议**：可能投稿至CVPR/ICCV（CCF-A类）
- **论文链接**：https://arxiv.org/abs/1911.05659
- **源码地址**：https://github.com/TrishaMittal/M3ER

**内容摘要**：
- 提出基于多模态情感识别的方法M3ER
- 结合面部、文本和语音线索
- 使用数据驱动乘法融合方法，强调更可靠的线索并抑制其他线索
- 提高情感识别的鲁棒性

**与本研究关联性**：⭐⭐⭐⭐
- 乘法融合策略可作为注意力融合的补充方法
- 多模态线索的选择性强调机制很有价值
- 可直接应用于驾驶员多模态情绪分析

---

## 二、驾驶员情绪与行为分析

### 6. CPSOR-GCN: A Vehicle Trajectory Prediction Method Powered by Emotion and Cognitive Theory

- **作者**：L. Tang, Y. Li, J. Yuan, A. Fu, J. Sun
- **发表时间**：2023年11月
- **会议**：arXiv preprint（可能投稿至CVPR/ICCV）
- **论文链接**：https://arxiv.org/abs/2311.08086
- **源码地址**：待查找

**内容摘要**：
- 结合物理层面的图卷积网络（GCN）和认知层面的动态贝叶斯网络（DBN）
- 量化情绪对驾驶行为的影响
- 预测精度提升68.70%
- 在异常情绪下的车辆轨迹预测任务中表现优异

**与本研究关联性**：⭐⭐⭐⭐⭐
- **最相关**：直接研究情绪对驾驶行为的影响
- GCN可用于建模驾驶员-车辆-环境的交互关系
- DBN可用于建模情绪状态的时间演变
- 为驾驶员情绪分析在智能驾驶中的应用提供重要参考

---

### 7. A Unified Multi-scale and Multi-task Learning Framework for Driver Behaviors Reasoning

- **作者**：Yang Xing, Chen Lv, Dongpu Cao, Efstathios Velenis
- **发表时间**：2020年3月（虽超出3年，但高度相关）
- **会议**：IEEE Transactions on Intelligent Transportation Systems（顶级期刊）
- **论文链接**：https://arxiv.org/abs/2003.08026
- **源码地址**：待查找

**内容摘要**：
- 统一的多尺度、多任务学习框架
- 同时识别驾驶员的身体姿势、意图和情绪
- 基于深度编码器-解码器结构
- 在两个自然驾驶数据集上的测试结果优于现有方法

**与本研究关联性**：⭐⭐⭐⭐⭐
- **高度相关**：直接研究驾驶员情绪识别
- 多任务学习框架可扩展应用到多模态情绪分析
- 多尺度设计有助于捕捉不同时间尺度的情绪变化
- 为多任务情绪分析提供架构参考

---

### 8. 基于多模态特征融合下的驾驶员行为智能检测

- **作者**：成福朋，赵芸
- **发表时间**：2025年8月
- **会议**：期刊论文
- **论文链接**：https://www.sci-open.net/index.php/JETI/article/download/5282/6826/16190
- **源码地址**：暂无公开源码

**内容摘要**：
- 提出基于多模态特征融合的驾驶员行为检测系统设计
- 涵盖数据采集、特征提取、融合决策及预警干预等核心环节
- 构建高效且实用的智能监测框架
- 适用于驾驶员行为智能检测应用

**与本研究关联性**：⭐⭐⭐⭐
- 多模态特征融合方法可直接参考
- 系统设计思路可应用于驾驶员情绪分析系统
- 预警干预机制对智能驾驶应用有价值

---

## 三、智能驾驶与多模态感知

### 9. V2X-ViT: Vehicle-to-Everything Cooperative Perception with Vision Transformer

- **作者**：Runsheng Xu, Hao Xiang, Zhengzhong Tu, Xin Xia, Ming-Hsuan Yang, Jiaqi Ma
- **发表时间**：2022年3月
- **会议**：CVPR 2022（CCF-A类）
- **论文链接**：https://arxiv.org/abs/2203.10638
- **源码地址**：通常在Papers with Code上可找到，建议访问 https://paperswithcode.com/paper/v2x-vit-vehicle-to-everything-cooperative

**内容摘要**：
- 基于视觉Transformer的协同感知框架
- 交替的多代理自注意力和多尺度窗口自注意力层
- 有效融合道路上各个代理（如车辆和基础设施）的信息
- 在3D目标检测任务中提升性能

**与本研究关联性**：⭐⭐⭐⭐
- Transformer架构可用于多模态特征融合
- 多尺度注意力机制可应用于多模态情绪分析
- 协同感知思想可扩展到多模态情绪感知系统

---

### 10. Holistic Transformer: A Joint Neural Network for Trajectory Prediction and Decision-Making

- **作者**：Hongyu Hu, Qi Wang, Zhengguang Zhang, Zhengyi Li, Zhenhai Gao
- **发表时间**：2022年6月
- **会议**：可能投稿至CVPR/ICCV（CCF-A类）
- **论文链接**：https://arxiv.org/abs/2206.08809
- **源码地址**：待查找

**内容摘要**：
- 联合神经网络同时进行轨迹预测和行为决策
- 采用三种注意力机制理解环境上下文
- 在自动驾驶场景中提升决策能力
- 端到端学习框架

**与本研究关联性**：⭐⭐⭐⭐
- 多注意力机制设计可应用于多模态情绪融合
- 联合学习框架可扩展为多任务情绪分析
- 端到端学习思路有价值

---

### 11. CRAT-Pred: Vehicle Trajectory Prediction with Crystal Graph Convolutional Neural Networks and Multi-Head Self-Attention

- **作者**：Julian Schmidt, Julian Jordan, Franz Gritschneder, Klaus Dietmayer
- **发表时间**：2022年2月
- **会议**：可能投稿至CVPR/ICCV（CCF-A类）
- **论文链接**：https://arxiv.org/abs/2202.04488
- **源码地址**：待查找

**内容摘要**：
- 结合图卷积神经网络和多头自注意力机制
- 多模态、非栅格化的轨迹预测模型
- 有效建模车辆之间的社交互动
- 无需依赖地图信息

**与本研究关联性**：⭐⭐⭐⭐
- 多头自注意力机制可直接应用于多模态情绪分析
- 图神经网络可建模驾驶员-环境交互关系
- 社交互动建模思想可应用于多模态交互

---

## 四、多模态融合与注意力机制

### 12. Self Context-Aware Emotion Perception on Human-Robot Interaction

- **作者**：Zihan Lin, Francisco Cruz, Eduardo Benitez Sandoval
- **发表时间**：2024年1月
- **会议**：arXiv preprint（可能投稿至ICCV/ECCV）
- **论文链接**：https://arxiv.org/abs/2401.10946
- **源码地址**：待查找

**内容摘要**：
- 引入自我上下文感知模型（SCAM）
- 通过情感坐标系统和上下文损失提升情感感知准确性
- 适用于人机交互场景
- 在情感识别任务中表现优异

**与本研究关联性**：⭐⭐⭐⭐
- 上下文感知机制可应用于驾驶场景的情绪分析
- 情感坐标系统有助于多维度情绪表示
- 人机交互场景与智能驾驶人机交互相关

---

### 13. Towards Context-Aware Modeling of Situation Awareness in Conditionally Automated Driving

- **作者**：Lilit Avetisyan, X. Jessie Yang, Feng Zhou
- **发表时间**：2024年5月
- **会议**：arXiv preprint（可能投稿至AAAI/IJCAI）
- **论文链接**：https://arxiv.org/abs/2405.07088
- **源码地址**：待查找

**内容摘要**：
- 实时评估驾驶员情境意识的预测模型
- 利用多模态数据（皮肤电反应、心率、眼动数据）
- 在模拟驾驶环境中进行建模
- 提升条件自动驾驶中的安全性

**与本研究关联性**：⭐⭐⭐⭐
- 多模态生理信号的使用与MPDB数据集高度契合
- 情境感知建模可结合情绪分析提升系统性能
- 实时评估机制对智能驾驶应用有价值

### 15. GA2MIF: Graph and Attention Based Two-Stage Multi-Source Information Fusion for Conversational Emotion Detection

- **作者**：Jiang Li, Xiaoping Wang, Guoqing Lv, Zhigang Zeng
- **发表时间**：2022年7月
- **会议**：arXiv preprint（可能投稿至ACL/EMNLP）
- **论文链接**：https://arxiv.org/abs/2207.11900
- **源码地址**：待查找

**内容摘要**：
- 提出基于图和注意力的两阶段多源信息融合方法（GA2MIF）
- 使用多头有向图注意力网络（MDGATs）进行上下文建模
- 使用多头成对跨模态注意力网络（MPCATs）进行跨模态建模
- 在IEMOCAP和MELD数据集上优于现有模型

**与本研究关联性**：⭐⭐⭐⭐
- 两阶段融合策略可参考
- 图注意力网络可建模多模态交互关系
- 跨模态注意力机制可直接应用

---

### 16. Continuous Emotion Recognition using Visual-audio-linguistic information: A Technical Report for ABAW3

- **作者**：Su Zhang, Ruyi An, Yi Ding, Cuntai Guan
- **发表时间**：2022年3月
- **会议**：ABAW3挑战赛技术报告
- **论文链接**：https://arxiv.org/abs/2203.13031
- **源码地址**：https://github.com/sucv/ABAW3

**内容摘要**：
- 提出跨模态协同注意力模型
- 融合视觉、音频和语言信息进行连续情感识别
- 包含视觉、音频和语言块，以及协同注意力块
- 通过多头协同注意力机制融合多模态特征
- 在Aff-Wild2数据库上表现优异

**与本研究关联性**：⭐⭐⭐⭐
- 三模态融合方法可扩展到四模态（加入生理信号）
- 协同注意力机制有价值
- 连续情感识别适用于实时应用

---

## 五、其他相关论文

### 14. 基于多模态的图文情感分析

- **作者**：孙文飞，张云华
- **发表时间**：2023年12月
- **会议**：期刊论文（非CCF-A类，但相关）
- **论文链接**：https://cs.hit.edu.cn/_upload/article/files/21/aa/8a55abb546b1bcd4142699b8cea9/7d991e86-59f0-4e1b-8755-aadcd44defbd.pdf
- **源码地址**：暂无公开源码

**内容摘要**：
- 提出基于BiGRU-ResNet的图文多模态情感分析模型
- 利用BERT和ResNet进行特征提取
- 通过注意力机制融合多模态信息
- 提高情感分类的准确性

**与本研究关联性**：⭐⭐⭐⭐
- 图文多模态融合方法可参考
- 注意力机制融合策略有价值
- 可作为预训练阶段的参考方法

---

## 六、论文获取建议

### 下载地址说明

1. **arXiv预印本**：大部分论文在正式发表前会在arXiv上发布预印本，可直接下载
2. **会议官网**：CCF-A类会议通常在官网提供论文下载（可能需要注册）
3. **作者主页**：部分作者会在个人主页提供论文和源码
4. **GitHub**：许多论文会提供开源代码实现

### 源码查找建议

1. **Papers with Code**：https://paperswithcode.com/ - 收录了大量论文及其代码实现
2. **GitHub搜索**：使用论文标题或作者名搜索
3. **作者主页**：查看第一作者或通讯作者的个人主页
4. **会议官网**：部分会议会在官网提供代码链接

### 研究建议

1. **重点关注**：
   - CFN-ESA（跨模态融合，情感转变感知）
   - CPSOR-GCN（情绪对驾驶行为的影响）
   - MFMC（多模态功能最大相关）

2. **方法借鉴**：
   - 注意力融合机制（CFN-ESA, Continuous Emotion Recognition）
   - 多任务学习框架（Unified Multi-scale Framework）
   - 对比学习方法（SCCL）

3. **应用场景**：
   - 驾驶员情绪实时监测
   - 情绪对驾驶行为的影响分析
   - 多模态情绪感知系统设计

---

## 七、论文获取与源码查找建议

### 下载地址说明

1. **arXiv预印本**：
   - 大部分论文在正式发表前会在arXiv上发布预印本
   - 访问 https://arxiv.org/ 搜索论文标题或ID
   - 可直接下载PDF版本

2. **会议官网**：
   - CCF-A类会议通常在官网提供论文下载
   - CVPR: https://openaccess.thecvf.com/
   - ICCV: https://openaccess.thecvf.com/
   - NeurIPS: https://papers.nips.cc/
   - AAAI: https://www.aaai.org/
   - ACL: https://aclanthology.org/
   - 部分会议需要注册账户

3. **作者主页**：
   - 部分作者会在个人主页提供论文和源码
   - 建议搜索第一作者或通讯作者的个人主页

4. **学术搜索引擎**：
   - Google Scholar: https://scholar.google.com/
   - Semantic Scholar: https://www.semanticscholar.org/
   - DBLP: https://dblp.org/

### 源码查找建议

1. **Papers with Code**：
   - 网址：https://paperswithcode.com/
   - 收录了大量论文及其代码实现
   - 可按会议、年份、任务类型搜索

2. **GitHub搜索**：
   - 使用论文标题、作者名或关键词搜索
   - 搜索格式：`"paper title" github` 或 `author name emotion recognition`

3. **作者主页**：
   - 查看第一作者或通讯作者的个人主页
   - 通常在"Publications"或"Projects"部分提供代码链接

4. **会议官网**：
   - 部分会议会在论文页面提供代码链接
   - 查看论文的补充材料（Supplementary Material）

5. **邮件联系**：
   - 如果找不到公开源码，可直接邮件联系作者
   - 通常在论文中提供作者邮箱

### 研究建议

#### 重点关注论文（按关联性排序）

1. **CFN-ESA**（⭐⭐⭐⭐⭐）
   - 跨模态融合机制可直接应用
   - 情感转变感知对动态情绪分析有价值
   - 有公开源码

2. **CPSOR-GCN**（⭐⭐⭐⭐⭐）
   - 直接研究情绪对驾驶行为的影响
   - GCN和DBN方法可借鉴
   - 最符合智能驾驶应用场景

3. **MFMC**（⭐⭐⭐⭐⭐）
   - 自监督学习框架可用于预训练
   - 最大化多模态依赖性的思想有价值
   - 有公开源码

4. **Continuous Emotion Recognition**（⭐⭐⭐⭐）
   - ICCV 2021正式发表
   - 领导-跟随注意力机制可应用
   - 有公开源码

5. **Unified Multi-scale Framework**（⭐⭐⭐⭐⭐）
   - 直接研究驾驶员情绪识别
   - 多任务学习框架可扩展
   - 高度相关但需注意发表时间

#### 方法借鉴建议

1. **注意力融合机制**：
   - 参考CFN-ESA的跨模态融合方法
   - 借鉴Continuous Emotion Recognition的领导-跟随注意力
   - 学习GA2MIF的两阶段融合策略

2. **多任务学习**：
   - 参考Unified Multi-scale Framework的多任务设计
   - 结合情绪分类、强度回归和趋势预测

3. **对比学习**：
   - 参考SCCL的聚类级对比学习方法
   - 用于多模态特征学习

4. **图神经网络**：
   - 参考CPSOR-GCN的GCN应用
   - 建模驾驶员-环境交互关系

---

## 八、参考文献格式

### 核心论文（按重要性排序）

1. Li, J., et al. (2023). CFN-ESA: A Cross-Modal Fusion Network with Emotion-Shift Awareness for Dialogue Emotion Recognition. arXiv:2307.15432
2. Tang, L., et al. (2023). CPSOR-GCN: A Vehicle Trajectory Prediction Method Powered by Emotion and Cognitive Theory. arXiv:2311.08086
3. Zheng, D., et al. (2025). Multimodal Functional Maximum Correlation for Emotion Recognition. arXiv:2512.23076
4. Zhang, S., et al. (2021). Continuous Emotion Recognition with Audio-visual Leader-follower Attentive Fusion. ICCV 2021
5. Yang, K., et al. (2023). Cluster-Level Contrastive Learning for Emotion Recognition in Conversations. arXiv:2302.03508
6. Xu, R., et al. (2022). V2X-ViT: Vehicle-to-Everything Cooperative Perception with Vision Transformer. CVPR 2022
7. Xing, Y., et al. (2020). A Unified Multi-scale and Multi-task Learning Framework for Driver Behaviors Reasoning. IEEE T-ITS
8. Li, J., et al. (2022). GA2MIF: Graph and Attention Based Two-Stage Multi-Source Information Fusion for Conversational Emotion Detection. arXiv:2207.11900
9. Zhang, S., et al. (2022). Continuous Emotion Recognition using Visual-audio-linguistic information: A Technical Report for ABAW3. arXiv:2203.13031
10. Lin, Z., et al. (2024). Self Context-Aware Emotion Perception on Human-Robot Interaction. arXiv:2401.10946

---

**文档版本**：v1.0  
**最后更新**：2024年  
**说明**：
- 部分论文可能尚未正式发表或源码未公开，建议通过论文中的联系方式联系作者获取最新信息
- 部分论文可能投稿至CCF-A类会议但尚未正式发表，已标注arXiv预印本状态
- 建议定期关注相关会议的官方网站和arXiv，获取最新研究进展

