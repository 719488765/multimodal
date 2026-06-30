# 毕业论文配图索引（SVG）

本目录为 **`毕业论文初稿框架.md`** 配套的**学术论文风格示意图**（矢量 SVG，便于 Word/LaTeX 插入后缩放不失真）。定稿时可将 SVG 转为 **EMF/PDF** 以满足部分学校对插图格式的要求。

| 编号 | 文件名 | 建议插入章节 | 内容说明 |
|------|--------|----------------|----------|
| 图 1-1 | `fig_01_01_research_framework.svg` | 第一章 绪论 | 研究总体框架：数据—模型—实验—智能体 |
| 图 3-1 | `fig_03_01_transformer_encoder.svg` | 第三章 相关知识 | Transformer 编码器子层结构示意 |
| 图 4-1 | `fig_04_01_data_pipeline.svg` | 第四章 数据与方案 | 多源数据治理与训练输入构造 |
| 图 4-2 | `fig_04_02_model_architecture.svg` | 第四章 | 多模态总体网络结构 |
| 图 5-1 | `fig_05_01_fusion_comparison.svg` | 第五章 融合设计 | 可切换融合策略与与参考学位论文叙事类比 |
| 图 5-2 | `fig_05_02_domain_adaptation.svg` | 第五章 | 域适应分支示意 |
| 图 6-1 | `fig_06_01_agent_architecture.svg` | 第六章 智能体 | emotion-agent 分层架构 |
| 图 6-2 | `fig_06_02_online_sequence.svg` | 第六章 | 在线滑窗推理时序 |
| 图 7-1 | `fig_07_01_experiment_protocol_ap.svg` | 第七章 实验 | AP0–AP4 与日志隔离 |

**整合总图（推荐答辩/绪论使用）**：

| 编号 | 文件名 | 说明 |
|------|--------|------|
| Figure 1 | [`../system_architecture_figure.html`](../system_architecture_figure.html) | **SVG 矢量图**（中文标注），横向流程图风格 |
| SVG 源文件 | [`../system_architecture_figure.svg`](../system_architecture_figure.svg) | 可直接插入 Word/LaTeX |
| 说明页 | [`index.html`](index.html) | 分图与总图对应关系、预览 |

**占位符在 Markdown 中的写法示例**：

```markdown
![图4-1 多源数据治理与训练输入构造流程](figures/thesis/fig_04_01_data_pipeline.svg)
```

**与吉林大学软件学院专硕 Word 模板的对应**：若模板中含「系统总体架构图」「模块结构图」「时序图」等，可优先映射 **图 6-1、图 6-2、图 4-2**；实验流程可映射 **图 7-1**。其余截图类（界面、TensorBoard）请在实验完成后自行导出为 **图 7-x 系列** 并登记于本表下方。

---

## 待补截图（实验完成后登记）

| 编号 | 内容 | 状态 |
|------|------|------|
| 图 7-2 | TensorBoard：val/accuracy 多 run 对比 | 【待补】 |
| 图 7-3 | TensorBoard：cls_ce_unweighted 监控 | 【待补】 |
| 图 7-4 | 混淆矩阵（混合验证） | 【待补】 |
| 图 8-1 | 智能体前端：情绪卡片 + 话术历史 | 【待补】 |
