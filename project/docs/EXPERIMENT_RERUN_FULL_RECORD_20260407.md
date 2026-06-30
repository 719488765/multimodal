# 重跑实验全量记录（更新至 logs_rerun 当前快照）

## 1. 文档目的

- 统一沉淀本次重跑实验的执行状态、关键指标、消融对比与阶段性结论。
- 数据以 `project/logs_rerun/*/metrics.csv` 为准；汇总可由 `scripts/summarize_rerun_results.py` 生成 `outputs_rerun/rerun_results_summary.*`。
- 标明异常与末段退化 run，避免论文中误用「末轮」或误读单次尖峰。

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

---

文档版本：**v3**  
数据快照：`project/logs_rerun/`（六组 `metrics.csv` 均已 50 条 val，统计时间：2026-04-11）  
上次全文更新：2026-04-11
