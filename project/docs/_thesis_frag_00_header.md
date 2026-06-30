# 毕业论文初稿（多源异构数据驱动的多模态情绪识别与情感智能体系统）

> **文档性质**：吉林大学软件学院专业硕士学位论文**初稿正文级**材料，用于在 Word 模板中分章迁移与排版。本文在结构上**对齐**钟谭媛《基于多模态特征融合的视频中人物情绪识别算法研究》之「**绪论—相关知识—算法设计—模型改进—系统应用—总结展望**」叙事逻辑，在**技术路线**上对齐本仓库 `project/`（训练与推理）与 `emotion-agent/`（在线采集与话术编排）的**真实实现**。  
> **图表说明**：矢量图见 `project/docs/figures/thesis/`，索引见同目录 `README_FIGURES.md`。文中以 `![图x-x …](figures/thesis/…)` 占位，Word 中可「链接到文件」或嵌入转换后的 EMF/PDF。  
> **数据口径**：凡涉及 **Acc/F1/epoch** 的定量表，以 `logs_*/*/metrics.csv` 与 `docs/THESIS_EXPERIMENT_MASTER_SUMMARY.md` 为准；尚未冻结的实验以 **【待补】** 标注，避免与终稿冲突。  
> **修订说明**：本版在既有「八章框架」基础上进行**大幅扩写**：充实摘要与绪论、扩展「相关知识」覆盖本项目所用全部主干技术、将「算法设计」拆解为数据方案与融合/优化两章并加长论述、扩展智能体系统工程章节、预置实验与系统测试写作骨架及与参考论文的对照说明。

---

## 图目录（初稿）

| 编号 | 题注 | 文件路径 |
|------|------|----------|
| 图 1-1 | 研究总体框架（算法—实验—系统） | `figures/thesis/fig_01_01_research_framework.svg` |
| 图 3-1 | Transformer 编码器子层结构示意 | `figures/thesis/fig_03_01_transformer_encoder.svg` |
| 图 4-1 | 多源数据治理与训练输入构造流程 | `figures/thesis/fig_04_01_data_pipeline.svg` |
| 图 4-2 | 多模态情绪识别总体网络结构 | `figures/thesis/fig_04_02_model_architecture.svg` |
| 图 5-1 | 可切换融合策略对比与叙事类比 | `figures/thesis/fig_05_01_fusion_comparison.svg` |
| 图 5-2 | 域适应分支与特征对齐示意 | `figures/thesis/fig_05_02_domain_adaptation.svg` |
| 图 6-1 | emotion-agent 分层架构 | `figures/thesis/fig_06_01_agent_architecture.svg` |
| 图 6-2 | 在线滑窗推理时序 | `figures/thesis/fig_06_02_online_sequence.svg` |
| 图 7-1 | AP0–AP4 实验序列与日志隔离 | `figures/thesis/fig_07_01_experiment_protocol_ap.svg` |
| 图 7-2～ | TensorBoard 曲线、混淆矩阵、系统界面 | 【实验完成后补录】 |

## 表目录（初稿）

| 编号 | 题注 | 所在章 |
|------|------|--------|
| 表 4-1 | 三数据集对比与本文角色 | 第四章 |
| 表 4-2 | 标签映射与统一类别空间（示意） | 第四章 |
| 表 5-1 | 融合策略与代码模块映射 | 第五章 |
| 表 7-1～7-6 | 主线重跑与 AP0–AP4 指标 | 第七章 |

---
