# 重跑实验全量记录（v4：logs_rerun + R4 + 中文微调）

## 1. 文档目的

- 统一沉淀本次重跑实验的执行状态、关键指标、消融对比与阶段性结论。
- **v3 及以前**：数据以 `project/logs_rerun/*/metrics.csv` 为准；汇总可由 `scripts/summarize_rerun_results.py` 生成 `outputs_rerun/rerun_results_summary.*`。
- **v4 起**：在保留 §3–§8 历史主线的前提下，追加 **SDAVT v3 R4（55 jobs）**、**MELD 中文微调 v1/v2**、**Agent Preset 选型**（§9–§14），与 `THESIS_EXPERIMENT_MASTER_SUMMARY.md` v2.0 对齐，供论文实验章与 emotion-agent 前端模型列表共用。
- 标明异常与末段退化 run，避免论文中误用「末轮」或误读单次尖峰；**不同日志桶禁止混排名**。

## 2. 数据来源与分析脚本

- 原始数据：`logs_rerun` 下各实验目录中的 `metrics.csv`（`phase=val` 按 epoch 汇总）。
- 脚本：
  - `scripts/summarize_rerun_results.py` → `outputs_rerun/rerun_results_summary.csv` / `.md`
  - `scripts/build_paper_table_main.py` → `outputs_rerun/paper_table_main.csv` / `.md`
  - `scripts/recompute_and_fill_paper_table.py`（checkpoint 映射齐全后可重算回填）
  - `scripts/recompute_val_metrics.py`（对指定 checkpoint 重算分类 Acc/F1，与 CSV 交叉验证）
- **重算状态**：定稿分类指标建议以 **CSV 峰值/末轮** 与 **checkpoint 重算** 双源一致为准；`paper_table_main_recomputed` 若仍为 partial，以本表及原始 `metrics.csv` 为准。

## 2.1 产物与原始日志一致性（当前 6 组）

- **执行状态（2026-04-11 快照）**：6 组实验 **均已满程**，验证集均为 **50 条记录（val epoch 0–49）**，可与「末轮」口径横向对比。
- `outputs_rerun/rerun_results_summary.csv` / `.md` 已用 `scripts/summarize_rerun_results.py` **刷新**，与本节 **§4** 及原始 `metrics.csv` 一致。

## 3. 主线 6 实验执行状态总览（与 tmux 会话对应）

| 会话（典型） | 实验设定 | 状态 | 备注 |
|---|---|---|---|
| `rerun_at_noda` | AT，无域适应 | **已完成** | val 0–49，50 条 |
| `rerun_at_da` | AT，域适应 | **已完成** | val 0–49 |
| `rerun_vt_noda` | VT，无域适应 | **已完成** | val 0–49 |
| `rerun_avt_da` | AVT，域适应 | **已完成** | val 0–49 |
| `rerun_avt_noda` | AVT，无域适应（standard 融合） | **已完成** | val 0–49；**末段 Acc/F1 塌陷**，见 §5.5 |
| `rerun_avt_es` | AVT，无域适应（emotion_shift） | **已完成** | val 0–49；**末轮 val loss 数量级异常**，见 §6.1 |

## 4. 核心结果汇总表（由 metrics.csv 逐条统计，客观值）

| Run（目录名关键词） | val 条数 / 范围 | 末轮 Acc | 末轮 F1 | 末轮 val loss | 最佳 Acc(轮次) | 最佳 F1(轮次) | 最低 val loss(轮次) |
|---|---|---:|---:|---:|---:|---:|---:|
| AT + noDA `...noDA_20260401_021657` | 50 / 0–49 | 0.254860 | 0.228168 | 1.878730 | 0.322354 (19) | 0.253779 (27) | 1.719402 (19) |
| AT + DA `...DA_20260403_134709` | 50 / 0–49 | 0.181965 | 0.134317 | 1.815104 | 0.352592 (24) | 0.312777 (24) | 1.737895 (24) |
| VT + noDA `...VT_...noDA_20260403_134709` | 50 / 0–49 | 0.373650 | 0.340640 | 1.751147 | 0.379590 (45) | 0.347355 (45) | 1.628153 (19) |
| AVT + DA `...AVT_...DA_20260403_134709` | 50 / 0–49 | 0.276458 | 0.226047 | 1.788666 | 0.353672 (42) | 0.270945 (42) | 1.672758 (42) |
| AVT + noDA（standard）`...noDA_20260407_193400` | **50** / **0–49** | **0.170086** | **0.123197** | **4.045835** | 0.372570 (27) | 0.328681 (27) | 1.594805 (23) |
| AVT + noDA（emotion_shift）`...emotion_shift_20260407_193400` | 50 / 0–49 | 0.374730 | 0.299816 | **56.703624** | **0.445464 (19)** | **0.403568 (19)** | 1.493677 (8) |

**说明**：上表末轮行对应各 run **epoch 49** 的 `phase=val` 记录。AVT standard 补满 50 epoch 后，**末轮 Acc/F1 仍接近 epoch 41 时的塌陷水平**（约 0.17 / 0.12），与 **Best@27（Acc≈0.37）** 反差极大，论文**不宜单独用末轮代表该配置能力**。

## 5. 消融对比与效果评价

### 5.0 数值差分速查（便于制表与答辩）

以下差分均由 §4 表中数值直接相减（**Last** = 末轮 epoch 49；**Best Acc** = 各 run 验证 Acc 最大值及出现 epoch）。

| 对比项 | Δ 末轮 Acc | Δ 末轮 F1 | Δ 最佳 Acc | 简要结论 |
|--------|------------|-----------|------------|----------|
| **AT+DA vs AT+noDA** | −0.072895 | −0.093851 | +0.030238 | 峰值升、末轮降 → 报 Best+Last |
| **VT+noDA vs AT+noDA** | +0.118790 | +0.112472 | +0.057236 | 引入视频模态收益明确 |
| **AVT+DA vs VT+noDA** | −0.097192 | −0.114593 | −0.025918 | 三模态+DA 未超 VT 本轮末轮/峰值 |
| **AVT+DA vs AVT+noDA（std）** | +0.106372 | +0.102850 | −0.018898 | DA 在 AVT 上抬高末轮，但峰值仍低于 emotion_shift |
| **emotion_shift vs AVT std（同配置除融合）** | **+0.204644** | **+0.176619** | **+0.072894** | 融合策略为本轮**最大增益来源之一**（末轮与峰值均优） |
| **emotion_shift vs VT+noDA** | +0.001080 | −0.040824 | +0.065874 | 末轮 Acc 与 VT 几乎持平；**峰值 Acc/F1 明显更高** |

### 5.1 DA 对 AT（AT+DA vs AT+noDA）

- **峰值**：AT+DA 最佳 Acc/F1 高于 AT+noDA（Acc +0.030238，F1 +0.058998，均出现在 epoch 24）。
- **末轮**：AT+DA 末轮 Acc/F1 低于 AT+noDA（Acc −0.072895，F1 −0.093851）。
- **评价**：域适应在本设置下抬高验证峰值，但末段不稳定或受域损失干扰，与「只报末轮」的写法容易冲突；论文建议 **同时报告 Best 与 Last**，并说明是否早停/调权。

### 5.2 引入视频（VT+noDA vs AT+noDA）

- 末轮 Acc 提升 **+0.118790**；最佳 Acc 提升 **+0.057236**。
- **评价**：本轮最清晰的**模态增益**来自 **VT**；与「轻量视频帧数 + 文本」配置相吻合。

### 5.3 三模态 + DA（AVT+DA vs VT+noDA）

- 末轮 Acc：AVT+DA 低于 VT+noDA（**−0.097192**）；最佳 Acc：略低（**−0.025918**）。
- **评价**：在当前超参与 batch 设置下，**三模态 + DA 未超越 VT 基线**；可作为「复杂度上升未必带来收益」的讨论点。

### 5.4 三模态无域适应：standard vs emotion_shift（核心消融）

- **峰值（论文亮点）**：emotion_shift 最佳 Acc **0.445464**、最佳 F1 **0.403568**（epoch 19），均高于 AVT noDA（standard）最佳 Acc **0.372570**、最佳 F1 **0.328681**（epoch 27）；也高于本轮 **AVT+DA** 的最佳 Acc/F1。
- **末轮（epoch 49）**：emotion_shift 末轮 Acc **0.374730**、F1 **0.299816**；与 VT+noDA 末轮 Acc **0.373650** 几乎持平（**+0.00108**），**明显高于** AVT standard 末轮 Acc **0.170086**（**+0.204644**）。
- **评价**：**emotion_shift 融合在本批 AVT noDA 上，峰值与末轮可用性均显著优于同配置 standard**，与项目内 CFN-ESA 类设计叙事一致；需结合 §6.1 对 **val total loss 数量级** 做严谨表述（分类指标与 loss 曲线分开解读）。

### 5.5 AVT noDA（standard）满程后的末段塌陷

- 当前日志已为 **50 个验证 epoch**；**epoch 49 末轮** Acc/F1 仍仅约 **0.17 / 0.12**，val loss 约 **4.05**，相对 AT/VT/AVT+DA 的末轮 loss（约 1.7–1.9）仍偏高。
- **epoch 27 附近峰值**仍为 Acc **0.372570**、F1 **0.328681**，与 emotion_shift 末轮 Acc 量级接近，说明模型能力曾在训练中期达到可用水平，**后期优化/校准退化**。
- **可能原因**（需结合训练日志核实）：学习率余弦末期震荡、小有效 batch 下噪声、ClassBalancedLoss 与 trend 头等耦合、或 checkpoint 选择若按 val loss 与按 F1 不一致等。
- **论文写法**：该 run 宜强调 **Best@27** 并报告 **Last@49** 说明不稳定；**禁止**仅用末轮与 VT/emotion_shift 末轮简单并列排名而不加说明。

## 6. 问题诊断、风险与可用性结论

### 6.1 AVT + emotion_shift 末轮 `val loss ≈ 56.7`（与 Acc 脱钩）

- 现象：末轮准确率仍约 **0.37**，但 **val loss** 达 **56.70**，与 AT/VT/AVT+DA（约 **1.7–1.9**）完全不在同一量级。
- **原因归纳（高置信）**：总损失为训练准则（含 **ClassBalancedLoss**）在验证集上的均值；按 batch 重算的类别权重会使 **加权 CE 与 argmax Acc 脱钩**；训练后期 **校准变差** 时亦可出现 **Acc 中等而 CE 极大**。旧版 CSV 中 val 的 `cls_loss` 列曾恒为 0（分项未写入），易误判；**新版训练**已将验证分项与 **不加权 CE** 落盘，新实验请以 `cls_loss` / `cls_ce_unweighted` 对照（见 `EXPERIMENT_ACCURACY_SEQUENCE_GUIDE.md`）。
- **建议**：写论文分类结果以 **Acc/F1** 与 **`recompute_val_metrics.py` 重算** 为准；loss 曲线单独讨论并引用 **分项/不加权 CE**。

### 6.2 横向对比公平性

- 六组均已 **50 epoch**，**末轮口径一致**；但 **AVT standard** 与部分 **DA** 实验仍呈现 **末轮远差于峰值**，对比时必须 **并列 Best 与 Last**。
- 不同 run 的模态、融合、`use_domain_adaptation` 等已体现在 `outputs_rerun/paper_table_main.csv` 等制表字段中，论文制表需一并注明 **batch、有效 batch、学习率、采样模式**（见各 `config/rerun/*.yaml`）。

### 6.3 模型可用性（阶段性结论）

| 用途 | 推荐 |
|---|---|
| **峰值能力展示 / 融合消融主结果** | AVT + noDA + **emotion_shift**（Best Acc/F1 本轮最高） |
| **末轮稳定、可演示部署候选** | **VT + noDA**（末轮与最佳相对均衡） |
| **论文基线锚点** | AT + noDA |
| **末轮 Acc 与 VT 接近、峰值更高** | AVT + emotion_shift（需说明 val total loss 口径） |
| **慎单独作「最终 SOTA」或仅报末轮** | AT+DA、AVT+DA（峰值与末轮反差大）；AVT noDA **standard**（末段塌陷，必报 Best@27） |

## 7. 综合结论（可直接压缩进论文「实验小结」）

1. **模态**：引入视频的 VT 相对 AT 提升明确；三模态 + DA 未超 VT，说明当前训练配方下 **复杂度与收益需再调**。
2. **域适应（AT）**：DA 提高验证峰值但末轮变差，属 **「有效但不稳」**，适合作为消融讨论而非唯一主模型。
3. **融合（AVT noDA）**：**emotion_shift 相对 standard 在峰值与末轮 Acc 上均显著更优**，支持「情感转变/跨模态融合」叙事；**standard 满 50 epoch 后末轮仍塌陷**，与 **Best@27** 反差大，写作时必须双指标报告。
4. **记录与可比性**：emotion_shift 的 **val total loss** 不宜与 AT/VT 的 loss 直接横比尺度；定稿分类指标建议 **CSV + checkpoint 重算** 双源一致；后续实验统一监控 **不加权 CE** 与分项损失。

## 8. 后续动作清单

1. ~~将 AVT noDA（standard）补跑至 50 epoch~~（**已完成**）；论文制表仍须 **Best@27 + Last@49** 并列说明末段退化。
2. 对 emotion_shift **best_f1（约 epoch 19）与 last（epoch 49）** checkpoint 运行 `recompute_val_metrics.py`，与 CSV 交叉验证后写入论文主表。
3. 刷新 `outputs_rerun`：`python3 scripts/summarize_rerun_results.py`（及按需 `build_paper_table_main.py`），使 summary 与 §4 一致。
4. 若需进一步提升混合准确率与消融完备度，按 `EXPERIMENT_ACCURACY_SEQUENCE_GUIDE.md` 在 **`logs_accuracy_seq`** 开展下一阶段实验（与本周 `logs_rerun` 隔离）。
5. **（v4）** 论文实验章主表已转向 **R4 单域 + 中文微调**；见下文 **§9–§12** 与 `THESIS_EXPERIMENT_MASTER_SUMMARY.md` §6.4 / §14。

---

## 9. v4 增量：实验时间线与文档角色（2026-04 → 2026-07）

| 阶段 | 日志桶 | 协议 | 文档锚点 | 论文用法 |
|------|--------|------|----------|----------|
| 主线重跑 | `logs_rerun/` | 三域混合 7 类 | 本文 §3–§8 | 模态/DA/融合早期消融 |
| 准确率序列 | `logs_accuracy_seq/` | 混合 + 单域上界 | `EXPERIMENT_ACCURACY_SEQ_*` | AP2 峰值、AP3 融合 |
| **SDAVT v3 R4** | **`logs_sdavt_v3_r4/`** | **单域** CREMA/MELD/MOSEI | `SDAVT_V3_R4_EXPERIMENT_RESULTS.md` + 本文 §10 | **学位论文实验主表** |
| 中文微调 | 同上 | MELD + 中文 BERT | 本文 §11 | Agent 部署 + 在线中文 |

**禁止混排名**：`logs_rerun` Best Acc≈0.45、AP2≈0.61、R4 MELD F1≈0.70 分属不同协议；论文须分表 + 脚注。

权威总控：`THESIS_EXPERIMENT_MASTER_SUMMARY.md` **v2.0**（含 Agent Preset §14）。

---

## 10. SDAVT v3 R4 全队列结果与消融（55 jobs，快照 2026-07-09）

**自动生成表**：[`SDAVT_V3_R4_EXPERIMENT_RESULTS.md`](SDAVT_V3_R4_EXPERIMENT_RESULTS.md)（`scripts/build_sdavt_r4_report.py`）。  
**日志**：`logs_sdavt_v3_r4/`；**权重**：`checkpoints_sdavt_v3_r4/`；**状态日志**：`outputs_sdavt_v3_r4/status/`；**TB**：`:6008`。

### 10.1 Phase 概览与冠军

| Phase | 目的 | 冠军 / 代表 | Best F1（或 Acc） | Agent Preset |
|-------|------|-------------|-------------------|--------------|
| p1_baseline | 单域基线 | R4_B_O0 (MOSEI) | F1 **0.6792** @12 | — |
| | | R4_B_C0 (CREMA) | F1 0.5889 @19 | — |
| | | R4_B_M1 (MELD) | F1 0.5680 @3 | — |
| p2_fusion | 融合策略 | F_O_ES / F_M_ES / F_C_ES | 见 §10.2 | `sdavt_mosei_r4`←F_O_ES |
| p3_c3 | CREMA 配方 | C3_C2_w2v_large | Acc **0.5672** / F1 0.5629 | `sdavt_crema_r4` |
| **p3_m3** | **MELD 配方** | **M3_M7_combo** | **F1 0.6957 @31 / Acc 0.7121** | **`sdavt_meld_v3_r4`** |
| p4_modal | 模态消融 | 见 §10.4 | 分域差异大 | 指导 skip_text / 骨干选型 |

### 10.2 融合消融（p2：emotion_shift vs standard 等）

| Dataset | ES Best F1 | STD Best F1 | Δ (ES−STD) | 其它要点 |
|---------|------------|-------------|------------|----------|
| MELD | F_M_ES **0.6109** | F_M_STD 0.4447 | **+0.166** | LFA/LFT/TS 均低于 ES |
| CREMA | F_C_ES **0.5786** | F_C_STD 0.2405 | **+0.338** | STD/TS 近崩溃；ES 必需 |
| MOSEI | F_O_ES **0.6792** | F_O_STD 0.5988 | **+0.080** | ES 仍优；与 p1 基线同量级 |

**与 `logs_rerun` 对照**：早期混合实验已得「emotion_shift ≫ standard」；R4 在**单域**上复现且 CREMA 上增益更大，适合作为论文**核心融合结论**的第二证据链。

### 10.3 MELD 配方消融（p3_m3，相对 M3_M0）

| Job | 改动要点 | Best F1 @ep | Δ vs M0 | 解读 |
|-----|----------|-------------|---------|------|
| M3_M0_baseline | ES 基线 | 0.6080 @3 | — | 锚点 |
| M3_M1_roberta | 文本→RoBERTa | **0.6823 @17** | **+0.074** | 文本骨干关键 |
| M3_M2_w2v_large | 音频 large | 0.5572 @3 | −0.051 | 单独换 large 未增益 |
| M3_M3_uniform | 均匀采样 | 0.6105 @3 | +0.002 | 边际 |
| M3_M4_focal | focal loss | 0.6079 @3 | ≈0 | 边际 |
| M3_M5_context | context | 0.5725 @3 | −0.036 | 未改善 |
| M3_M6_moddrop | modality dropout | 0.6079 @6 | ≈0 | 边际 |
| **M3_M7_combo** | **组合（含 RoBERTa 等）** | **0.6957 @31** | **+0.088** | **英文主轨 / 论文冠军** |

训练日志目录：`logs_sdavt_v3_r4/SDAVT_R4_M3_M7_combo/`；best ckpt：`checkpoints_sdavt_v3_r4/SDAVT_R4_M3_M7_combo/checkpoint_pretrain_best_f1.pth`。  
CSV 核对（2026-07-16）：Best F1=**0.695750** Acc=**0.712094** @ep31；Last（ep32）F1=0.688 Acc=0.706。

### 10.4 模态消融（p4_modal）要点

| Dataset | 强组合（Best F1 量级） | 崩溃/极弱 | 对 Agent 含义 |
|---------|------------------------|-----------|---------------|
| MELD | T / AT / VT / AVT ≈0.67–0.68 | V alone collapse | 英文对话依赖文本 |
| MOSEI | T **0.7087** 最强；VT/AVT/AT 高 | — | 文本主导 |
| CREMA | V / AV / AVT ≈0.33–0.35 | T / A / AT collapse | 表演语料文本极弱 → 在线可 `leader_audio` / skip_text |

### 10.5 CREMA / MOSEI 代表结果（完整表见自动报告）

- **CREMA p3**：C3_C1 0.5336 → C3_C2 **0.5629** → C3_C3 0.5526（focal 未超 C2）。  
- **MOSEI**：p1/p2 ES 与基线同为 **0.6792**；p4 纯文本可达 **0.7087**（模态上界讨论用，非部署默认）。

---

## 11. 中文微调轨（接在 M3_M7 之后）

### 11.1 实验设计

| 项目 | v1 `chinese_agent` | v2 `chinese_agent_v2`（全量） |
|------|--------------------|-------------------------------|
| Config | `M3_M7_chinese_agent.yaml` | `M3_M7_chinese_agent_v2.yaml` |
| 文本骨干 | **bert-base-chinese** | 同左 |
| 初始化 | 自 M3_M7（英文）迁移 / 中文词表适配 | **自 v1 best_f1**（`Partial checkpoint loaded … loaded=1031`） |
| 数据 | MELD（英文字幕进中文分词器） | MELD + **500×`*_zh.txt` ASR 伪标签** + **97 agent_capture** |
| `max_train_samples` | 全量 | **0（全量）**；此前 smoke=256 已归档 |
| 训练样本数（日志） | — | train subset **10085** / val **1108**（meld） |
| 状态日志 | `outputs_sdavt_v3_r4/status/m3m7_zh_finetune.log` | `.../m3m7_zh_v2_full_finetune.log` |
| Preset | `sdavt_meld_zh_agent` | **`sdavt_meld_zh_agent_v2`（默认）** |

### 11.2 指标对比（`metrics.csv` phase=val）

| Run | Best F1 @ep | Best Acc | Last F1 @ep | 早停 |
|-----|-------------|----------|-------------|------|
| M3_M7_combo（英文参照） | **0.6957 @31** | **0.7121** | 0.688 @32 | — |
| chinese_agent **v1** | **0.6010 @9** | 0.6273 | 0.5973 @13 | 有（日志 28 行级） |
| chinese_agent **v2 全量** | **0.6114 @5** | **0.6363** | 0.6020 @10 | **patience=5，best 恢复 ep5** |

v2 训练过程摘要（`m3m7_zh_v2_full_finetune.log`）：

- Epoch 0：val_f1=0.6017 → 保存 best_f1  
- Epoch 5：val_f1=**0.6114** val_acc=**0.6363** → 最终 best  
- Epoch 10：early stopping；`Restored best-F1 checkpoint from epoch 5`；`EXIT_CODE=0`  
- Checkpoint：`checkpoints_sdavt_v3_r4/SDAVT_R4_M3_M7_chinese_agent_v2/checkpoint_finetune_best_f1.pth`

### 11.3 消融分析（论文可写段落要点）

1. **换中文 BERT 相对英文 M3_M7：ΔF1 ≈ −0.095**  
   监督标签与对话文本仍主要来自英文 MELD；词表切换改善「在线中文分词/embedding」，但离线英文验证集会下降——**论文必须分表**：「离线英文 MELD」vs「中文部署轨」。
2. **v2 vs v1：ΔF1 ≈ +0.010**  
   中文伪字幕 + agent 采集注入带来稳定增益；早停在 ep5，说明增强数据在初期即起作用，后期易过拟合。
3. **在线 E2E（2026-07-16）**：中文句「我很难过」→ sad conf≈**0.534**（preset=v2），证明部署轨与离线 F1 同向可用。
4. **不建议**对 MOSEI/CREMA 做同类中文微调（无中文监督；CREMA 文本模态本就崩溃）——与 Master §14.3 一致。

### 11.4 与早期中文实验的关系

| 轨 | 位置 | 说明 |
|----|------|------|
| `agent_chinese`（AP2 配方） | `logs_accuracy_seq` / preset 同名 | 遗留；**新部署优先 v2** |
| R4 中文 v1/v2 | `logs_sdavt_v3_r4` | **当前主线** |

---

## 12. 跨阶段消融总表（论文「结果与讨论」压缩用）

| 科学问题 | 证据桶 | 关键定量结论 | 写作注意 |
|----------|--------|--------------|----------|
| 视频模态价值 | `logs_rerun` | VT vs AT：末轮 Acc **+0.12** | 混合协议 |
| 域适应 | `logs_rerun` + AP4 | 峰值升、末轮常降 | Best+Last |
| 融合 emotion_shift | `logs_rerun` + **R4 p2** | 混合 Best Acc +0.07；R4 MELD F1 **+0.17**；CREMA **+0.34** | 双证据链 |
| 混合配方 | AP2 | Best Acc ~**0.61** | 与 R4 分表 |
| 文本骨干 / 配方 | **R4 p3_m3** | RoBERTa +0.074；combo **+0.088→0.696** | 论文主表 |
| 模态贡献 | **R4 p4** | MELD/MOSEI 文本强；CREMA 文本弱 | 支撑 Agent 策略 |
| 中文适配 | **§11** | v1 0.601→v2 **0.611**；相对英文 −9.5pt | 部署 vs 论文分轨 |

---

## 13. emotion-agent Preset 选型速查（与 Master §14 同步）

| 场景 | Preset | Best F1 | 默认？ |
|------|--------|---------|--------|
| 中文在线 Agent | `sdavt_meld_zh_agent_v2` | 0.6114 | **是**（`.env`） |
| 英文对话 / 论文 MELD | `sdavt_meld_v3_r4` | 0.6957 | 英文自动路由 |
| 中文消融对照 | `sdavt_meld_zh_agent` | 0.6010 | 否 |
| 混合三域演示 | `ap2_m1` | ≈0.56 | 否 |
| MOSEI/CREMA 实验 | `sdavt_mosei_r4` / `sdavt_crema_r4` | 0.679 / Acc0.567 | experimental |

语言自动路由：zh→v2，en→`sdavt_meld_v3_r4`（见 `chinese_inference_router.py`）。

---

## 14. v4 综合结论（可并入论文实验小结）

1. **`logs_rerun`（v3）**：模态、DA、emotion_shift 的早期证据仍有效；standard 末段塌陷须 Best+Last。  
2. **R4（55 jobs）**：单域协议下确认 **ES 融合**、**M3_M7 为 MELD 冠军（F1=0.696）**、**模态依赖分域差异**——应作为学位论文**主实验表**。  
3. **中文微调**：仅在 MELD 上做中文 BERT；v2 全量达 **F1=0.6114**，定为 Agent 默认；离线英文与在线中文**分表报告**。  
4. **工程**：Preset 列表以 `config.py` / Master §14 为 UI 选项权威源；后续改权重须同步改 metadata 数字与本文 §11/§13。

---

文档版本：**v4**  
数据快照：  
- `project/logs_rerun/`（六组满程，统计 2026-04-11）  
- `project/logs_sdavt_v3_r4/`（R4 done=55，自动报告 2026-07-09；中文 v2 全量 2026-07-15，`EXIT_CODE=0`）  
上次全文更新：**2026-07-16**  
关联主控文档：`THESIS_EXPERIMENT_MASTER_SUMMARY.md` **v2.0**
