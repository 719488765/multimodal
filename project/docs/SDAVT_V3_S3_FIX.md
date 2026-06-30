# SDAVT v3 S3 修复方案（S2 失败后）

> 2026-06-16：S2 三域均早停；根因已定位并修复。

## 1. S2 结果回顾

| Run | Best Acc | Best F1 | 早停@ep | 症状 |
|-----|----------|---------|---------|------|
| S2-M1 MELD | **0.626** | **0.605** | 29 | **成功** — 早停正常，非失败 |
| S2-C1 CREMA | 0.193 | 0.096 | 13 | train loss≈1.79≈ln(6)，随机猜测 |
| S2-O0 MOSEI | 0.233 | 0.088 | 10 | **全 epoch 指标逐位相同** |

## 2. 根因

### 2.1 MOSEI — NaN 塌缩（非早停问题）

1. **Focal Loss + 长序列 T=100**：反向传播产生 NaN logits；`nan_to_num` 掩盖后 loss 锁死在 **1.4296**
2. **`set_backbone_frozen` 误解冻 BERT**：`unfreeze_encoder_layers=0` 时 epoch 1 仍开启 BERT 梯度 → 数值爆炸
3. **解冻后未重建 optimizer**：npy 投影层虽解冻但不在 optimizer 参数组

验证：关闭 Focal + 保持 BERT 冻结后，20 step 内 loss 3.1→0.36。

### 2.2 CREMA — 融合策略错误（非早停问题）

1. **emotion_shift + use_text=false**：text 全零参与加权融合与 cross-attn，logits_std≈0.04，无法区分 6 类
2. **native 标签**：neutral id 5 vs unified id 4，与 S1 口径不一致
3. S1-C0（AVT+emotion_shift）Acc 0.336 > S2-C1（AV）0.193 — 去掉 text 反而更差

验证：`fusion_strategy: standard` + AV，15 step loss 2.0→1.75，logits_std 0.5。

### 2.3 MELD — 早停属正常

- Best @ep19：Acc 0.626 / F1 0.605，超过 M1 目标（0.58/0.54）
- ep29 触发 patience=10 早停，best ckpt 已保存

## 3. S3 配置

| ID | 文件 | 关键改动 |
|----|------|----------|
| S3-O0 | `S3_O0_mosei_AVT_ES_npy_ap1fix.yaml` | ClassBalanced β=0.9999；无 Focal；unified 标签；monitor val_accuracy；patience 12 |
| S3-C0 | `S3_C0_crema_AV_standard_ap1fix.yaml` | standard 融合；AV；unified；batch 2；freeze 0 |

## 4. 代码修复

- `train.py`：`set_backbone_frozen` 尊重 `unfreeze_encoder_layers`；解冻时 `build_training_optimizer` 重建
- `balanced_loss.py`：Focal `pt` clamp 防溢出

## 5. 启动

```bash
bash scripts/start_sdavt_v3_s3_tmux.sh all
bash scripts/start_sdavt_v3_s3_tmux.sh eval_meld   # S2 最优 ckpt 评估
bash scripts/start_sdavt_v3_s3_tmux.sh clean_tb    # 归档 S2 日志
```

## 6. 验收

- S3-O0 MOSEI：val Acc **≥0.70**；指标随 epoch 变化（非冻结）
- S3-C0 CREMA：val Acc **≥0.35**（阶段性）
- MELD：沿用 S2 best_f1 ckpt，eval 后更新 Agent preset
