# SDAVT v3 S2 优化方案（MELD / CREMA / MOSEI 重训）

> 针对 S1 三域基线未达标的根因分析与 S2 对策。

## 1. 结果回顾

| Run | Best Acc | Best F1 | 目标 | 主要症状 |
|-----|----------|---------|------|----------|
| S1-M1 MELD | 0.509 @ep14 | 0.427 @ep17 | 0.58 / 0.54 | **neutral 召回 91%**；sadness/fear/disgust 接近 0 |
| S1-C0 CREMA | 0.336 @ep23 | 0.322 @ep18 | 0.50 / 0.45 | 各类均衡但整体低；train/val loss 同步偏高（欠拟合） |
| S1-O0 MOSEI | ~0.233 | ~0.088 | 0.73 / 0.62 | **val 指标多 epoch 完全相同**；远低于 AP1 ep0 Acc 0.716 |

对照 AP1 历史：**MELD Acc 0.580 @ep36**（无早停、50 epoch）；CREMA Acc ~0.28–0.34；**MOSEI Acc 0.716 @ep0–2**（npy 特征易拟合，WD 1e-5）。

---

## 2. 根因分析

### 2.1 MELD（S1-M1 v2 配方）

| 维度 | S1-M1 实际 | AP1 成功配置 | 影响 |
|------|-----------|-------------|------|
| 标签空间 | unified（surprise→anxious, disgust→other） | 同 unified | 语义映射噪声，少数类更难学 |
| dropout | **0.25** | 0.1 | 融合层过度正则，欠拟合 |
| weight_decay | **1e-4** | 1e-5 | 10× 更强 L2，抑制收敛 |
| 视频输入 | 8×224 | **4×112** | 更大输入但未增加有效 epoch |
| 损失 | ClassBalanced β=0.9999 | ClassBalanced（但 train loss 口径不同） | weighted loss ~2.0 vs CE ~1.4，优化 landscape 更陡 |
| 早停 | patience **3**, max 20 ep | **无早停**, 50 ep | best @ep17，AP1 best @**ep36** — 停太早 |
| Text BERT | unfreeze 2 层 | 全冻结 | 额外可训练参数易扰动 |
| 类别分布 | val neutral **42%** | 同 | 模型塌缩到 neutral |

**结论**：v2 配方（高 dropout / 强 WD / 短训早停）与 AP1 收敛路径不兼容；应 **回归 AP1 超参 + native 标签 + Focal 损失** 处理少数类。

### 2.2 CREMA（S1-C0 AVT）

| 维度 | 问题 |
|------|------|
| **Text 模态** | ASR 转写为演员朗读的**中性句子**（如 "It's eleven o'clock"），与 6 类情感**无语义相关**，BERT 引入噪声 |
| leader_modal | audio（合理）但 text 仍参与融合 |
| 损失 | ClassBalanced 对**已均衡**的 6 类无必要 |
| 视频 | 8×224 + WD 1e-4 vs AP1 4×112 + 1e-5 |

**结论**：CREMA 应改为 **AV（无 text）** + AP1 超参 + plain CE。

### 2.3 MOSEI（S1-O0 AVT npy）

| 维度 | S1-O0 实际 | AP1 成功配置 | 影响 |
|------|-----------|-------------|------|
| weight_decay | **1e-4** | **1e-5** | 过强正则，抑制 npy 特征快速拟合 |
| 损失 | ClassBalanced | plain CE / 无 CB | 与 val neutral **72%** 偏态冲突 |
| batch | 4 × grad2 | 1 × grad2 | 有效 batch 更大但非主因 |
| 标签 | native 7 类 | unified 7 类 | native 下 val 仅 4 类有样本（0,2,3,4） |
| 症状 | Acc/F1 **epoch 0–4 完全相同** | ep0 Acc 0.716 | 疑似优化停滞或恒预测类 |

**结论**：回归 **AP1 超参**（WD 1e-5、Focal γ=2、dropout 0.1）；保留 native + npy AVT 管线；50ep patience 10。

---

## 3. S2 重训配置

| ID | 数据集 | 配置 | 关键改动 |
|----|--------|------|----------|
| **S2-M1** | MELD | `S2_M1_meld_AVT_ES_native_ap1plus.yaml` | native 7 类；4×112；dropout 0.1；WD 1e-5；Focal γ=2；50ep patience 10 |
| **S2-C1** | CREMA | `S2_C1_crema_AV_ES_no_text.yaml` | **use_text: false**；leader audio；plain CE；4×112；50ep patience 8 |
| **S2-O0** | MOSEI | `S2_O0_mosei_AVT_ES_npy_ap1plus.yaml` | npy V+A+SDK T；WD 1e-5；Focal γ=2；batch 2 grad4；50ep patience 10 |

### 启动

```bash
bash scripts/start_sdavt_v3_s2_tmux.sh all      # GPU0 MELD+MOSEI + GPU1 CREMA
bash scripts/start_sdavt_v3_s2_tmux.sh mosei    # 仅 MOSEI S2-O0
bash scripts/start_sdavt_v3_s2_tmux.sh clean_tb   # 归档 S1 日志，TB 仅 S2
```

### 验收目标

- S2-M1：Val Acc **≥0.55**（native 空间）；macro-F1 **≥0.48**；neutral recall **<80%**
- S2-C1：Val Acc **≥0.40**（阶段性；CREMA 全数据集 historically ~0.35）
- S2-O0：Val Acc **≥0.70** @ep0–5（对齐 AP1）；macro-F1 **≥0.60**；指标应随 epoch 变化（非冻结）

### 当前进度（2026-06-14）

| Run | Epoch | Val Acc | Val F1 | 备注 |
|-----|-------|---------|--------|------|
| S2-M1 | 2 | 0.551 | 0.491 | 优于 S1；继续训练 |
| S2-C1 | 6 | 0.156 | 0.043 | 仍低，ep10+ 再判 |
| S2-O0 | 0 | — | — | 刚启动 |

---

## 4. 后续矩阵（S2 仍不达标时）

- MELD：roberta-base text；weighted sampler；S1-M0 AP1 精确复现（unified 对照）
- CREMA：VT / VA 消融；更大 batch；wav2vec 微调
- MOSEI：S2-O1 AVT vs AT 消融；排查 S1 指标冻结（dataloader shuffle、loss 梯度、head 初始化）
