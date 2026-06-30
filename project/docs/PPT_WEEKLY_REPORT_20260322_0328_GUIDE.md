# 工作总结汇报 PPT 编写指导（3月22日–3月28日）

编写说明：本文档对齐 `PPT_WEEKLY_REPORT_20260315_0321_GUIDE.md` 的版式，面向本周（2026-03-22 至 2026-03-28）。本周汇报仅围绕实验工作：AVT_noDA 结果与消融定位、`val/precision` 指标事故与修复、AVT_DA 推进及与无 DA 对照；不展开项目文档本身的整理优化。

依据来源：`PROJECT_OVERVIEW_AND_TRAINING_PLAN.md` 第 10.5.7 节（AVT noDA）、第 8.6 / 9.5 节中的视频扩展与消融叙事；`logs/AVT_pretrain_3datasets_noDA_20260323_202809`、`logs/AVT_pretrain_3datasets_DA_20260325_215401`；代码侧 `scripts/train.py`、`utils/helpers.py`、`scripts/recompute_val_metrics.py`。

数据摘录说明：下文 第五节、第六节（附录 A / B） 及文中带 「logs 实测」 的数值，均来自本仓库 `project/logs/<run>/metrics.csv` 与 `project/checkpoints/` 的当前磁盘快照（解析方式：`phase=val` 按 epoch 汇总；训练行为 `phase=train` 的 `loss`）。CSV 中的 precision / recall / f1 仍可能受历史口径影响，与 `recompute_val_metrics.py` 不一致时以重算为准（见第 3 页）。

---

## 一、本周工作范围与时间线（实验向）

| 日期 | 内容概要 |
|------|----------|
| 3月22日–3月24日 | 复核 AVT_noDA 全量 run 与论文主表口径；对照主文档 10.5.7 整理「与 AT / VT / V-only」的消融层次 |
| 3月24日–3月25日 | 验证集 precision/recall/F1 计算链路排障与修复：定位 `train.py` 中 `calculate_metrics` 入参顺序错误；加固 `helpers.calculate_metrics`；同步 `recompute_val_metrics.py`；对 AVT_noDA 等历史日志与 checkpoint 重算对照 |
| 3月25日起 | 启动 AVT + DA 三数据集混合预训练：`config/config_AVT_DA.yaml`，日志 `AVT_pretrain_3datasets_DA_20260325_215401` |
| 3月25日–3月28日 | AVT_DA 长时训练 + TensorBoard 监控（如端口 6007）；截至本周末验证已记录至 epoch 41（50 epoch 配置下仍在跑） |

PPT 可直接使用（时间线 bullet）  
- 本周主线：AVT_noDA 结论固化 + 指标口径纠错 → AVT_DA 对照实验启动。  
- 工程收获：验证指标与训练脚本对齐后，后续实验与论文数字以「修复后代码 + 重算脚本」为准，避免再次误读 precision/F1。

---

## 二、PPT 整体结构建议（实验专用）

建议页序：封面 → 本周实验概述 → val/precision 问题专项（原因·修改·解决·重跑策略）→ AVT_noDA 与既有消融对比 → AVT_DA 阶段性进展与和无 DA 对照 → 实验经验 → 下周计划。

叙事主线：在 同一 `MultimodalEmotionModel`、配置驱动消融 前提下，先厘清 全模态无 DA（AVT_noDA） 在证据链中的位置，再在同一视频轻量设定下推进 AVT_DA，回答「域适应是否改善三模态跨数据集混合学习」。

---

## 三、各页内容与占位

### 第 1 页：封面

- 标题：工作总结汇报  
- 副标题：李智春 | 2026年3月22日–3月28日  
- 可选副标题：AVT 无 DA 基线固化 + 验证指标修复 + AVT 域适应实验推进中  

---

### 第 2 页：本周实验概述

PPT 可直接使用文案  
- AVT_noDA（已完成，主文档 10.5.7）：run `logs/AVT_pretrain_3datasets_noDA_20260323_202809`，`config/config_AVT_noDA.yaml`，50 epoch 跑满，作为论文「Audio+Video+Text、无域适应」主结果行。  
- 指标事故处理：发现历史训练中 `val/precision`（及连带 F1）与真实预测不一致；完成代码修复与 checkpoint 重算 流程，部分依赖旧 CSV 的结论需以重算值或选择性重跑为准。  
- AVT_DA（本周启动，进行中）：run `AVT_pretrain_3datasets_DA_20260325_215401`，与 AVT_noDA 同融合（standard）、同轻量视频、同 batch_size=1，仅 打开域适应模块与域损失，用于 noDA vs DA 消融。  
- 监控：TensorBoard + `metrics.csv/jsonl`；长时用 tmux 防断连。

---

### 第 3 页：val/precision 指标错误——原因、修改、解决与重跑（本周重点）

PPT 可直接使用文案（建议分四层小标题）

1）现象  
- 同一 checkpoint 用 `recompute_val_metrics.py`（修复后） 重算的 precision / F1，与当时 `metrics.csv` / TensorBoard `val/precision` 可能出现 明显偏差；accuracy 往往仍与重算 一致或接近（因 `accuracy_score` 对 (y_true, y_pred) 与交换顺序在「逐样本相等」意义下对称，不暴露顺序错误）。

2）根因（代码层面，高置信度）  
- `scripts/train.py` 的 `validate()` 中曾将 `calculate_metrics` 的两个位置参数顺序写反（把 targets 当作 predictions 传入，或反之）。  
- `sklearn.metrics.precision_recall_fscore_support` 第一个参数为真实标签 y_true，第二个为预测 y_pred；顺序颠倒后，precision、recall、F1 的统计含义错位，而 accuracy 仍可能「看起来合理」，最易误判为「模型 precision 异常高/异常低」。  
- 另：部分 epoch 段 CSV 出现 precision≈1.0 等形态，还与 多类不平衡、加权平均方式、以及当时 `labels` 未显式固定 等叠加有关；主文档 10.5.7 已提示 Epoch 0–32 附近不宜单独作为论文最终 precision 结论，应以修复后重算或后段 epoch 为准。

3）修改与解决（工程动作）  
- `utils/helpers.py::calculate_metrics`：分类任务 强制 1D、转 int；按 预测与标签并集推断类别数 并 `labels=np.arange(num_classes)` 传入 `precision_recall_fscore_support`，减少「缺类」时的口径漂移；空集与长度不匹配显式处理。  
- `scripts/train.py`：改为 `calculate_metrics(predictions=all_preds, targets=all_targets)` 关键字调用，杜绝后续维护时再误传顺序。  
- `scripts/recompute_val_metrics.py`：同样关键字调用，便于对 任意 checkpoint 在固定 val 上复现指标。  

4）对已完成实验的影响与重跑策略  
- 论文主表 / 周报数字：对 AVT_noDA 终点（如 epoch 49） 及关键对照点，应用 当前代码 + `recompute_val_metrics.py` 输出 accuracy / precision / recall / f1，写入 10.5.7 补充行或脚注。  
- 是否整实验重跑：非必须整段 50 epoch 重跑——若仅需 PR/F1 可信，重算优先；若需 TensorBoard 曲线与 CSV 全文历史 完全自洽，可 选择性重跑 或保留旧 run 仅作 loss 趋势参考、指标以重算表为准。  
- AVT_DA：建议在 修复后代码 上继续或收官，新 run 的 CSV 与旧 AVT_noDA 部分 epoch 段不可混比 precision 绝对值，比趋势时注明口径。

---

### 第 4 页：AVT_noDA 实验总结（与既有消融对比）

配置与 run（与主文档 10.5.7 一致）  
- 目录：`logs/AVT_pretrain_3datasets_noDA_20260323_202809`  
- 配置：`config/config_AVT_noDA.yaml` — `use_audio=true, use_video=true, use_text=true`；无 DA；`fusion_strategy=standard`；`batch_size=1`；视频 frame_size=112，num_frames=4。  
- 命令：`python3 scripts/train.py --config config/config_AVT_noDA.yaml --mode pretrain`  

效果评价（摘自 10.5.7 + logs 实测，汇报时 precision/F1 建议附「已重算」脚注）  
- 训练（`metrics.csv`，`phase=train` 的 `loss`）：epoch 0 → 49：2.526353 → 2.254212（run `AVT_pretrain_3datasets_noDA_20260323_202809`），与 AT 终点量级可比但优化更难。  
- 验证（混合 val，同 run CSV）  
  - 终点 epoch 49：`val/loss_total` 3.014232，accuracy 0.166307，precision 0.470465，recall 0.166307，f1 0.239977（*precision/F1 建议重算后写入论文主表*）。  
  - CSV 内最高 accuracy：同上 epoch 49（0.166307）。  
  - CSV 内最高 f1：epoch 3，f1 0.273193，accuracy 0.158207（*不宜单独作最终结论，见 10.5.7 与第 3 页口径说明*）。  
  - 最小 val loss：epoch 0，2.718054（仅作曲线参考，不代表最优分类性能）。  

与之前实验的消融对比（PPT 可直接贴表：主文档口径 + 本仓库 logs 实测终点）  

| 设置 | 代表 run（`logs/`） | 终点 val（epoch，CSV） | Acc / F1（CSV） | 一句话结论 |
|------|---------------------|-------------------------|-----------------|------------|
| AT noDA | `AT_pretrain_3datasets_noDA_20260305` | 49 | 0.136069 / 0.183260 | 本快照终点低于历史最优；CSV 内 最高 Acc 0.196544（epoch 35） |
| AT + DA | `AT_pretrain_3datasets_DA_20260312_201945` | 49 | 0.143629 / 0.138437 | DA 分支可训；终点 F1 未高于本表中 AT noDA 终点（需结合重算与多 seed 再论） |
| T-only | `T_pretrain_3datasets_noDA_20260311` | 49 | 0.160907 / 0.176114 | CSV 内 最优 Acc 在 epoch 0：0.215443，F1 0.273432 |
| A-only | `A_pretrain_3datasets_noDA_20260311` | 49 | 0.080994 / 0.149850（precision 列 1.0，疑为指标形态/口径问题） | 单音频仍弱；A-only 建议重算 precision |
| V-only | `V_only_pretrain_3datasets_noDA_20260317_205923` | 49 | 0.086393 / 0.128497 | 单视频信号弱 |
| VT noDA | `VT_pretrain_3datasets_noDA_20260321_142141` | 49 | 0.082073 / 0.121832 | 本 CSV 仅含 val epoch 15–49（续训段）；epoch 34 最佳 Acc 0.116631，F1 0.188889 |
| AVT noDA | `AVT_pretrain_3datasets_noDA_20260323_202809` | 49 | 0.166307 / 0.239977 | 全模态无 DA 主行；终点 Acc 高于本表 AT noDA 终点，但 低于 AT 的 epoch 35 峰值 |
| AVT_DA | `AVT_pretrain_3datasets_DA_20260325_215401` | 41（未满程） | 0.158207 / 0.138673 | 进行中；CSV 内最佳 Acc 0.164147（epoch 23），最佳 F1 0.140287（epoch 39） |

经验小结（PPT bullet，与上表 logs 实测对齐）  
- 相对 VT（续训段 CSV）：AVT_noDA 终点 F1 0.240 高于 VT 最佳 F1 0.189（epoch 34） 与 终点 F1 0.122，说明在现有设定下 补回音频对三模态仍有实质帮助。  
- 相对 AT noDA（本快照 CSV）：AVT_noDA 终点 Acc 0.166 高于 AT_noDA 终点 Acc 0.136，但 低于 AT_noDA 峰值 Acc 0.197（epoch 35）；论文句避免笼统写「AT 一定高于 AVT」，应区分 终点 vs 峰值 并 重算对齐。  
- AVT_DA vs AVT_noDA（DA 未满程）：当前 DA 最后 val（epoch 41） Acc/F1 低于 noDA 终点（epoch 49）；不下最终结论，待 50 epoch + best checkpoint 重算 后再写对照。  

图片占位：TensorBoard 中 `AVT_pretrain_3datasets_noDA_20260323_202809` 的 `val/accuracy`、`val/loss_total`、`train/loss_classification`。

---

### 第 5 页：AVT_DA 阶段性进展（与 AVT_noDA 对照）

设定（保证可比）  
- 配置：`config/config_AVT_DA.yaml` — 在 AVT_noDA 基础上 `model.domain_adaptation.enabled=true` 且 `training.loss.use_domain_adaptation=true`（`domain_loss_weight` 等见 yaml）。  
- Run：`logs/AVT_pretrain_3datasets_DA_20260325_215401`  
- 进度（logs 实测）：`metrics.csv` 中 `phase=val` 共 42 条（val epoch 0–41），train 已至 epoch 41（50 epoch 未满程）。  
- 训练总 loss（`phase=train`）：epoch 0 → 41：2.567025 → 2.493142。  

阶段性观察（摘自该 run 的 `metrics.csv`，趋势参考；关键数字定稿前建议重算）  
- 域分支：train 中 domain 相关损失在 epoch 9–12 附近出现 尖峰（约 0.56–0.69），之后回落至约 0.18–0.22，符合域对抗常见拉锯。  
- 验证（CSV）：`val/accuracy` 由早期约 0.04–0.12 抬升；epoch 23：Acc 0.164147，F1 0.071585；epoch 39：Acc 0.163607，F1 0.140287（CSV 内 F1 最高）；epoch 41（当前最后一条 val）：`val/loss_total` 3.242273，Acc 0.158207，Prec 0.196677，Rec 0.158207，F1 0.138673。  
- 与 AVT_noDA 的阶段性对比（谨慎表述）：在 未满程、且 val 指标需重算口径统一 的前提下，本周只宜写 「DA 分支已稳定接入、曲线可训练」；是否优于 noDA 需 同 epoch 检查点重算 + 跑满 50 epoch 后比 best。  

图片占位：TensorBoard 并列 noDA run 与 DA run 的 `val/accuracy`、`train/loss_domain`（或等价标量）。

---

### 第 6 页：实验经验总结（可放「教训 + 资产」）

PPT 可直接使用文案  
- 消融叙事统一：所有结果均在 同一模型骨架 + YAML 开关 下完成，论文表头写清 模态 / 是否 DA / 融合策略 / 视频规格，避免不可比。  
- 指标可信链：accuracy 不能单独证明 pipeline 正确；precision/recall/F1 必须用 y_true/y_pred 顺序正确 + 必要时重算；重要结论 双源校验（CSV + `recompute_val_metrics.py`）。  
- AVT 工程：batch_size=1 + 轻量视频 是当前 GPU 与现实数据下的 务实折中；坏视频仍会拖慢 epoch，需与指标问题区分排障。  
- 下一步资产：AVT_noDA 提供 全模态无 DA 锚点；AVT_DA 提供 域适应变量；两者对齐后形成 8.6.2 中的「视频 + DA」证据链。

---

### 第 7 页：阶段性效果评级（实验项）

| 项目 | 状态 | 评级 | 说明 |
|------|------|------|------|
| AVT_noDA 全量预训练 | 已完成（10.5.7） | B | 论文消融价值高；precision 等需重算后写主表 |
| val 指标链路修复 | 已完成 | A- | 降低后续实验误判风险 |
| AVT_DA 预训练 | 进行中（至 val epoch 41） | B-（阶段性） | 待满程与 noDA 同口径对比 |
| 消融证据链（AT/VT/AVT） | 已贯通 | B+ | 支撑「模态与难度」论述 |

结论句  
- 本周在 不夸大 DA 效果 的前提下，完成了 全模态无 DA 基线的论文级定位 与 指标可信性修复，并把 AVT_DA 推进到可对照分析的中间节点。

---

### 第 8 页：下周工作计划（3月29日–4月4日建议）

1. 跑满 AVT_DA（50 epoch），保存 `checkpoint_pretrain_best.pth` 与周期 checkpoint。  
2. 对 AVT_noDA（epoch 49）与 AVT_DA（best / epoch 49） 统一执行：  
   `conda run -n myenv310 python3 scripts/recompute_val_metrics.py --config <对应 yaml> --checkpoint <path>`，更新论文主表与 10.5.7 / 新增 DA 小节。  
3. 撰写对照结论：在同一重算口径下回答 「AVT_DA 相对 AVT_noDA 是否带来稳定增益」；若无增益，分析 域损失权重、lambda、数据噪声。  
4. 按主文档 8.6.2 / 9.10，择一推进：AVT_noDA_emotion_shift（结构消融）或 清洗数据后短程复验 AVT（可选）。  
5. 时间允许：T_only / A_only 短跑，补全模态消融表。  

---

## 四、AVT_DA 数据速查（`metrics.csv`，趋势用）

> 路径：`logs/AVT_pretrain_3datasets_DA_20260325_215401/metrics.csv`（val 共 42 条，epoch 0–41）

| 阶段 | val epoch | accuracy | f1 | 备注 |
|------|-----------|----------|-----|------|
| 早期 | 0–8 | 约 0.037–0.120 | 约 0.009–0.045 | 冷启动 |
| Acc 峰值（CSV） | 23 | 0.164147 | 0.071585 | 与 noDA 终点 Acc 接近 |
| F1 峰值（CSV） | 39 | 0.163607 | 0.140287 | 未满程下的 F1 高点 |
| 当前最后一条 val | 41 | 0.158207 | 0.138673 | `val/loss_total` 3.242273 |

---

## 五、附录 A：`checkpoints/` 目录快照与使用说明

> 以下文件名与修改时间来自本仓库 `project/checkpoints/` 当前列表；`checkpoint_pretrain_*.pth` 会被最近一次 `--mode pretrain` 覆盖，引用时需结合 时间戳 + 日志 run 名 判断归属。

| 文件 | 体量（约） | 修改时间（日志摘录） | 说明 |
|------|------------|----------------------|------|
| `checkpoint_pretrain_epoch_49.pth` | 1.2G | 3月25日 18:29 | 时间与 `AVT_pretrain_3datasets_noDA_20260323_202809` 的 `metrics.csv` 一致，极可能为 AVT_noDA 满程最后一档 |
| `checkpoint_pretrain_epoch_44.pth` | 1.2G | 3月25日 13:04 | AVT_noDA 训练过程中定期保存 |
| `checkpoint_pretrain_epoch_39.pth` … `epoch_4.pth` | 各 1.2G | 3月26日–3月28日 | 与 AVT_DA（3/25 晚启动）时间线连续，当前多为 DA 实验写入 |
| `checkpoint_pretrain_best.pth` | 1.2G | 3月26日 00:36 | 当前为「最近一次 pretrain 的 val loss 最优」；若需严格对应某次 run，应以 TensorBoard / 该 run 的 metrics.csv 中 best epoch 交叉验证 |
| `checkpoint_finetune_*.pth` | 约 1016M | 3月14日 | CREMA 微调系列，与本周 AVT 主线独立 |

建议：论文归档时对关键 run 另存为 `checkpoint_AVT_noDA_epoch49.pth` 等 带实验名 的文件，避免被后续 pretrain 覆盖。

---

## 六、附录 B：关键 `logs/*/metrics.csv` 验证集摘要（全表可复算）

> 下列为 `phase=val` 自动汇总；precision/recall/f1 以重算脚本为准（见第 3 页）。

| `logs/<run>/` | val 条数 | val epoch 范围 | 最后 val：loss / acc / f1 | val 最佳 acc（epoch） | val 最佳 f1（epoch） |
|---------------|----------|------------------|----------------------------|------------------------|------------------------|
| `AT_pretrain_3datasets_noDA_20260305` | 50 | 0–49 | 3.945343 / 0.136069 / 0.183260 | 0.196544（35） | 0.267540（6） |
| `AT_pretrain_3datasets_DA_20260312_201945` | 50 | 0–49 | 4.412652 / 0.143629 / 0.138437 | 0.169006（16） | 0.258409（6） |
| `T_pretrain_3datasets_noDA_20260311` | 50 | 0–49 | 4.435868 / 0.160907 / 0.176114 | 0.215443（0） | 0.273432（0） |
| `A_pretrain_3datasets_noDA_20260311` | 50 | 0–49 | 8.262451 / 0.080994 / 0.149850 | 0.154428（1） | 0.267540（1） |
| `V_only_pretrain_3datasets_noDA_20260317_205923` | 50 | 0–49 | 4.662072 / 0.086393 / 0.128497 | 0.090713（13） | 0.139628（0） |
| `VT_pretrain_3datasets_noDA_20260321_142141` | 35 | 15–49 | 3.436951 / 0.082073 / 0.121832 | 0.116631（34） | 0.188889（34） |
| `AVT_pretrain_3datasets_noDA_20260323_202809` | 50 | 0–49 | 3.014232 / 0.166307 / 0.239977 | 0.166307（49） | 0.273193（3） |
| `AVT_pretrain_3datasets_DA_20260325_215401` | 42 | 0–41 | 3.242273 / 0.158207 / 0.138673 | 0.164147（23） | 0.140287（39） |

训练总 loss（`phase=train` 首末 epoch）  

| run | epoch 0 train loss | 末 epoch train loss |
|-----|--------------------|---------------------|
| `AT_pretrain_3datasets_noDA_20260305` | 2.497362 | 2.244927（49） |
| `AVT_pretrain_3datasets_noDA_20260323_202809` | 2.526353 | 2.254212（49） |
| `AVT_pretrain_3datasets_DA_20260325_215401` | 2.567025 | 2.493142（41） |

---

## 七、图片与 TensorBoard 占位

| 页面 | 建议插图 |
|------|----------|
| 第 4 页 | `AVT_pretrain_3datasets_noDA_20260323_202809`：val acc / loss、train cls loss |
| 第 5 页 | 并列 noDA 与 `AVT_pretrain_3datasets_DA_20260325_215401`：val acc；DA run 的 domain 相关曲线 |
| 第 3 页（可选） | 同 checkpoint CSV precision vs 重算 precision 一行对比表（举证指标修复） |

---

## 八、编写注意事项

1. precision 事故页建议单独成页，便于答辩时说明「不是模型玄学，是评估代码顺序错误 + 后续已修复」。  
2. AVT_noDA 数字：口头汇报可引用 10.5.7；书面材料 precision/F1 以重算为准。  
3. AVT_DA：未满程不写「已证实 DA 有效」，写 阶段性观察与下周对照计划。  
4. 与上周 PPT（3.15–3.21）衔接：上周侧重 V-only / VT / scheduler 断崖；本周侧重 全模态 AVT 基线 + 指标可信 + DA 启动。  

---

文档版本：v3（补充 `logs/`、`checkpoints/` 实测摘录表；消融表与 CSV 对齐）  
对应周次：2026-03-22 至 2026-03-28  
关联文档：`docs/PROJECT_OVERVIEW_AND_TRAINING_PLAN.md`（10.5.7、8.6、9.5）；`config/config_AVT_noDA.yaml`；`config/config_AVT_DA.yaml`  
数据路径：`project/logs/`、`project/checkpoints/`  
