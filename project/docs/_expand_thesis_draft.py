#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Expand 毕业论文初稿框架.md: new header, long abstract, ch1 enrich, ch3 supplement, figure refs."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "毕业论文初稿框架.md"
HDR = ROOT / "_thesis_frag_00_header.md"
OUT = ROOT / "毕业论文初稿框架.md"

NEW_ABSTRACT = r"""## 摘要

随着智能座舱、辅助驾驶与人机共驾技术的普及，车辆不再仅承担「位移工具」功能，而逐步演化为可感知乘员状态的移动交互空间。驾驶员与前排乘员的情绪状态——如愤怒、焦虑、悲伤、平静等——与注意力分配、接管意愿、对系统提示的信任度等行为因素存在广泛关联；因此，**稳定、可解释、可工程落地的情绪感知能力**成为下一代智能汽车人机交互（Human–Machine Interaction, HMI）链条中的关键一环。与传统单模态识别相比，**多模态情绪识别**能够同时利用面部与肢体视觉线索、语音韵律与声学特征、以及文本语义信息，在光照变化、遮挡、车内噪声、方言与语速变化等复杂条件下提供互补证据，从而提升鲁棒性。

然而，面向「论文级系统闭环」的研究仍面临三类相互耦合的困难。第一，**数据层面的异质性与域偏移**：公开情感数据集在采集场景（演播室表演、电视剧对白、网络视频评论）、标注粒度（离散类别 vs 连续效价/激活度）、类别空间（例如 6 类与 7 类并存）等方面差异显著；若缺乏统一协议与清洗治理，混合训练易导致模型学到「数据集捷径特征」，表现为验证集 **峰值准确率与末轮准确率严重背离**、不同损失项与 Argmax 准确率走势不一致等。第二，**模型层面的结构选择与对照可比性**：跨模态注意力、情感转变建模、非对称主导模态融合、两阶段由粗到细融合等机制在文献中各有动机，但若工程实现缺少统一骨架与配置开关，消融实验容易退化为「不可比堆叠」。第三，**工程层面的可复现性与系统落地**：训练日志、断点续训状态、媒体健康检查、推理端与训练端配置一致性等问题，会直接决定研究结论是否可信，也决定算法能否接入在线采集—推理—话术生成—可视化闭环。

本文工作围绕「**统一模型骨架 + 配置驱动实验 + 准确率优化序列 + 情感智能体原型**」展开，对应参考学位论文中「**算法设计—模型改进—系统应用**」的主线，但在模态与任务设定上从「**视频：面部表情 + 肢体动作**」拓展为「**音—视—文**」三模态，并在工程形态上以 **BERT/Wav2Vec2/ResNet** 等预训练骨干替代传统手工特征 + SVM 流水线，以 **YAML 单一事实源** 管理实验变量。具体而言：在算法层面，构建了 `MultimodalEmotionModel`，以 **ResNet50** 提取短视频片段视觉表征，以 **Wav2Vec2-base** 提取定长音频波形表征，以 **BERT-base-uncased** 提取文本上下文表征；在统一隐空间内实现 **标准跨模态注意力融合（standard）**、**情感转变感知融合（emotion_shift，借鉴 CFN-ESA 思想）**、**领导—跟随式融合（leader_follower）** 与 **两阶段融合（two_stage）**；并可按需启用 **域对抗/域适应**、**类别不平衡损失（如 Class-Balanced / Focal）**、**功能最大相关（FMC）思想的可选约束** 等。在数据与实验层面，整合 **CREMA-D、MELD、CMU-MOSEI** 三数据集：完成主线 **六组重跑**（`logs_rerun/`，覆盖 AT/VT/AVT 与域适应、融合策略等关键对照），并设计与实施 **准确率优化序列 AP0～AP4**（`logs_accuracy_seq/`，与历史重跑目录强制隔离），从 **单域上界、三混合配方、融合结构、域损失扫描** 等维度系统扫描性能与稳定性；实验记录与主汇总见 `docs/THESIS_EXPERIMENT_MASTER_SUMMARY.md`。在系统层面，实现 **emotion-agent** 工程原型：以 **FastAPI** 作为后端网关，串联 **本地 ASR（Whisper 兼容接口）**、**情绪推理服务**、**大语言模型话术编排** 与 **React + Vite 前端**，形成可演示的「采集—推理—交互—归档」闭环，并对 **配置—检查点一致性** 给出可操作的绑定清单。

在方法论贡献的表述上，本文强调：不仅报告「最佳验证指标」，同时报告 **末轮指标** 与 **`cls_ce_unweighted` 等解耦监控量**，以解释 **ClassBalancedLoss 等 batch 级重加权** 可能造成的 **训练总损失与分类准确率脱钩**；该做法与 `MIXED_DATASET_TRAINING_ANALYSIS.md` 的问题分析一致，可作为学位论文「**实验可信度与评价口径**」小节的实质内容。当前阶段，AP0～AP3 已形成可写入初稿的表格化结果；AP4 仍部分进行中，智能体端到端性能指标（延迟分位数、稳定性、主观满意度）待系统测试完成后补齐。【待补】在摘要末段以 2～3 句给出**最终定量结论句**（含混合验证 Best/Last、与单域上界差距一句、系统演示效果一句），并与英文 Abstract 严格对译。

**关键词**：多模态情绪识别；跨模态注意力；Transformer；BERT；Wav2Vec2；ResNet；域适应；类别不平衡；配置驱动实验；情感智能体；智能汽车人机交互

---

"""

CH1_INSERT_AFTER_11 = r"""
![图1-1 研究总体框架（算法—实验—系统）](figures/thesis/fig_01_01_research_framework.svg)

**图 1-1 说明**：上图概括了本文「数据治理—模型训练—消融矩阵—系统部署」四层闭环。与钟谭媛论文以 **FBER→AM-C3D→AM-FBER** 串起「特征提取—注意力增强—系统验证」类似，本文以 **统一 MultimodalEmotionModel** 串起 **多模态编码—可切换融合—可选域适应—智能体服务化**；差别在于本文更强调 **跨数据集混合训练** 与 **工程可复现日志**，而非单一 FABO 数据集上的精度迭代。

### 1.5 研究对象、术语与论文边界

本文所称「情绪」默认指 **离散类别标签空间** 下的分类结果（以配置 `emotion_classes` 为主，混合验证为 7 类；CREMA 单域实验为 6 类并需脚注）。连续情绪维度（效价/激活度）在代码中可按配置启用，但**主线实验表格以分类指标为主**。「驾驶员/乘员」在公开数据集实验中由演员或视频人物代理；**不在缺乏专有数据支持时声称完成真实道路驾驶员大规模标注与验证**。

### 1.6 与参考学位论文的结构对照（写作提示）

钟谭媛论文第二章为 **C3D、注意力、稀疏编码树、字典学习** 等；第三章为 **FBER 算法流程**；第四章为 **AM-C3D**；第五章为 **AM-FBER + 原型系统**；第六章为 **总结展望**。本文将「相关知识」扩展为 **Transformer/BERT/Word2Vec 对照、Wav2Vec2、ResNet、评价指标、数据治理通论** 等，以覆盖三模态深度实现；将「算法设计」拆解为 **第四章（数据与总体方案）** 与 **第五章（融合与优化/AP 序列）** 两章，以对应「**数据集与流程** + **模型改进与训练策略**」；将系统章节对应 **第六章智能体** 与 **第七章实验/测试** 分置，便于与学院 Word 模板中「系统总体设计 / 系统实现 / 系统测试」栏目映射。

"""

CH3_SUPPLEMENT = r"""
## 3.13 优化器、学习率调度与「有效 batch」概念（与 AP2 实验解释强相关）

深度网络训练普遍采用 **随机梯度下降类优化器**。本文默认使用 **Adam / AdamW**（以各 `yaml` 的 `training.optimizer` 为准），其优势在于对学习率尺度相对不敏感、可自适应调整各参数步长。实践中，**batch_size** 与 **gradient_accumulation_steps** 共同决定 **有效 batch（effective batch）**；在显存受限时，小物理 batch 会带来梯度高方差，进而影响混合域训练的稳定性。`logs_accuracy_seq` 中 **AP2-M1（effbatch8）** 的改进，应在论文中以「**优化噪声—域对齐难度**」视角解释，而非仅写「调参变好」。

学习率策略常见 **Warmup + Cosine/Linear decay**。Warmup 有助于避免训练早期大学习率破坏预训练骨干表征；Cosine 衰减有助于中后期细致搜索更平坦极小值附近。论文写作应摘录关键超参：`learning_rate`、`warmup_epochs`、`max_grad_norm` 等，并说明与 **seed=3407** 的可复现组合。

## 3.14 批归一化、Dropout 与正则化通论

卷积与全连接层常配合 **BatchNorm** 以缓解内部协变量偏移、加速收敛；BERT 等 Transformer 内部使用 **LayerNorm** 而非 BatchNorm。Dropout 通过随机屏蔽神经元抑制过拟合；在多模态场景中，亦可对某一模态特征以一定概率丢弃以模拟 **模态缺失**，提升推理端降级能力。L2 权重衰减（weight decay）在 AdamW 中与解耦。本文若启用 **Label Smoothing**，应在论文中说明其对校准与类别混淆的影响。

## 3.15 混合精度训练、梯度裁剪与数值稳定性（如启用）

在 GPU 上启用 **Automatic Mixed Precision（AMP）** 可降低显存占用并提高吞吐，但可能引入数值敏感问题；若训练配置启用 AMP，论文应声明并记录 **loss scaling** 相关现象。`max_grad_norm` 梯度裁剪用于抑制梯度爆炸，尤其在 **RNN/LSTM 生理分支** 或 **对抗训练域分支** 并存时更值得报告。

## 3.16 数据增强与媒体质量：工程视角的「数据即模型一部分」

对视频常见增强包括随机裁剪、颜色抖动、水平翻转（需注意左右语义不对称时禁用翻转）。对音频可引入噪声注入、时间遮蔽（SpecAugment 思想）等。本文主线更强调 **坏媒体剔除与读取鲁棒性**（`check_media_health_dir.py` 等），其工程收益可能超过轻量随机增强：在跨数据集训练中，**异常样本会导致 dataloader 阻塞与有效训练时间骤降**，从而在论文「实验环境」中形成可写段落。

## 3.17 Word2Vec 与 BERT 的对比实验设计建议（可选附录）

若篇幅允许，可增加小规模对照：**冻结词嵌入 + 浅层编码器** vs **BERT 微调**，在相同融合模块与相同数据子集上比较。该实验不是为了否定 Word2Vec，而是为论文提供「**静态分布式表示 vs 上下文表示**」在跨数据集情绪语义上的实证材料，呼应第三章理论分析。

## 3.18 PyTorch 工程要点：Dataset、DataLoader 与随机性

`Dataset.__getitem__` 应保证返回张量形状与类型稳定；`DataLoader` 的 `collate_fn` 在多源混合 batch 时需处理 **变长文本 padding**、**视频帧数对齐**、**音频长度对齐** 等。随机性来源包括：数据顺序打乱、dropout、部分 CUDA 算子非确定性；论文应说明是否使用 `torch.backends.cudnn.deterministic` 等策略，并解释 **重复实验方差** 来源。

## 3.19 注意力机制的多重语义：从 CBAM 到跨模态 Transformer

参考论文将 **CBAM** 与 **3D 卷积** 结合形成 **3DCBAM**，强调通道与空间维度的重新加权。本文在视频侧以 **ResNet50** 为主干，可在讨论中将 **SE/CBAM** 作为「**与参考论文改进路线对照**」的可选扩展；在融合侧则以 **多头注意力** 实现跨模态交互，二者数学形式均属于「**对特征进行重加权/重组合**」，可在论文中用一小节做概念统一，避免读者认为「注意力」一词在不同章节指不同对象。

## 3.20 本章与后续章节的接口关系（写给答辩用的「路线图」）

第三章负责建立 **符号、概念与训练工程通论**；第四章把这些概念落到 **三数据集协议与目录结构**；第五章落到 **融合/域适应/AP 序列** 的可切换实现；第六章落到 **在线系统**；第七章用 **日志事实** 回答「是否有效、在何种条件下有效、为何不稳定」。这一分工与 `THESIS_FULL_DRAFT_MULTIMODAL_EMOTION.md` 中强调的「**证据链闭环**」一致：方法—实现—日志—系统四位一体。

"""

FIG34 = r"""
![图4-1 多源数据治理与训练输入构造流程](figures/thesis/fig_04_01_data_pipeline.svg)

![图4-2 本文多模态情绪识别总体网络结构示意](figures/thesis/fig_04_02_model_architecture.svg)

"""

FIG5 = r"""
![图5-1 可切换融合策略对比与叙事类比](figures/thesis/fig_05_01_fusion_comparison.svg)

![图5-2 域适应分支与特征对齐示意](figures/thesis/fig_05_02_domain_adaptation.svg)

"""

FIG6 = r"""
![图6-1 emotion-agent 分层架构](figures/thesis/fig_06_01_agent_architecture.svg)

![图6-2 在线滑窗推理时序](figures/thesis/fig_06_02_online_sequence.svg)

"""

FIG7 = r"""
![图7-1 准确率优化序列 AP0–AP4 与日志目录隔离](figures/thesis/fig_07_01_experiment_protocol_ap.svg)

"""


def main() -> None:
    import re

    base = SRC.read_text(encoding="utf-8")
    hdr = HDR.read_text(encoding="utf-8")

    idx_abs = base.find("## 摘要")
    if idx_abs < 0:
        raise SystemExit("## 摘要 not found")
    base = hdr.strip() + "\n\n" + base[idx_abs:]

    old_abs = re.search(
        r"## 摘要\n\n.+?\*\*关键词\*\*：.+?\n\n---\n",
        base,
        flags=re.DOTALL,
    )
    if not old_abs:
        raise SystemExit("could not find abstract block")
    base = base[: old_abs.start()] + NEW_ABSTRACT + base[old_abs.end() :]

    # Insert CH1 figure after "## 1.1 研究背景与意义" first paragraph - after first double newline block
    needle = "完整科研训练过程。\n\n## 1.2"
    if needle in base and "![图1-1" not in base:
        base = base.replace(needle, "完整科研训练过程。\n\n" + CH1_INSERT_AFTER_11 + "## 1.2", 1)

    # Fig 3-1 after "## 3.5 Transformer" heading
    n2 = "## 3.5 Transformer 与自注意力机制\n\n### 3.5.1"
    if n2 in base and "![图3-1" not in base:
        base = base.replace(
            n2,
            "## 3.5 Transformer 与自注意力机制\n\n"
            + "![图3-1 Transformer 编码器子层结构示意](figures/thesis/fig_03_01_transformer_encoder.svg)\n\n"
            + "### 3.5.1",
            1,
        )

    # Ch3 supplement before "## 3.12 本章小结" - insert before 3.12
    m12 = "## 3.12 本章小结\n"
    if m12 in base and "## 3.13" not in base:
        base = base.replace(m12, CH3_SUPPLEMENT + m12, 1)

    # Fig 4 after "## 4.4 总体模型结构概述"
    n4 = "## 4.4 总体模型结构概述\n\n`MultimodalEmotionModel`"
    if n4 in base and "![图4-1" not in base:
        base = base.replace(
            n4,
            "## 4.4 总体模型结构概述\n\n" + FIG34 + "`MultimodalEmotionModel`",
            1,
        )

    # Fig 5 after "# 第五章" first paragraph end - after 5.1 title block - insert after "## 5.1 问题描述" section header + first para
    n5 = "## 5.1 问题描述：混合训练中的冗余与失配\n\n三数据集混合时"
    if n5 in base and "![图5-1" not in base:
        base = base.replace(
            n5,
            "## 5.1 问题描述：混合训练中的冗余与失配\n\n" + FIG5 + "三数据集混合时",
            1,
        )

    # Fig 6 after "## 6.2 总体架构"
    n6 = "## 6.2 总体架构\n\n推荐分层"
    if n6 in base and "![图6-1" not in base:
        base = base.replace(n6, "## 6.2 总体架构\n\n" + FIG6 + "推荐分层", 1)

    # Fig 7 after "## 7.1 实验环境与复现性"
    n7 = "## 7.1 实验环境与复现性\n\n【待补】"
    if n7 in base and "![图7-1" not in base:
        base = base.replace(n7, "## 7.1 实验环境与复现性\n\n" + FIG7 + "【待补】", 1)

    # Footer revision
    base = base.replace(
        "**修订记录（初稿框架）**：第 3 批已扩写原第二章 §2.5～§2.6；第 4 批已插入独立「第二章 国内外研究现状」、原第二～七章顺延为第三～八章，并撰写满页级英文 Abstract；第七章已嵌入主记录文档表格；已增加 **致谢** 与 **附录 C**。",
        "**修订记录（初稿框架）**：第 5 批（本版）在八章结构下完成 **初稿级大幅扩写**：重写加长中文摘要；绪论增列结构对照与边界说明；第三章增列训练工程与注意力概念补充；第四～七章嵌入 `figures/thesis/` 矢量配图占位；图表目录独立登记。实验定量以 `THESIS_EXPERIMENT_MASTER_SUMMARY.md` 为锚，终稿前请再次核对 metrics。",
    )

    OUT.write_text(base, encoding="utf-8")
    print("written", OUT, "chars", len(base))


if __name__ == "__main__":
    main()
