# P3 防过拟合重训 (2026-06-29)

## 问题

`M3_M1_roberta` / `M3_M7_combo` 在 ep5 backbone 解冻后验证 F1 从 ~0.686 跌至 ~0.44（严重过拟合）。

## 代码修复 (`scripts/train.py`)

- **ResNet 视频 backbone 永久冻结**（与 Wav2Vec2 一致），仅训练 fusion + head + RoBERTa 末层
- **Modality dropout**：训练时按 `modality_dropout` 随机丢弃整模态（M3_M7）
- **训练结束恢复 `checkpoint_pretrain_best_f1.pth` 权重**

## 配置调整

| 参数 | 旧值 | 新值 |
|------|------|------|
| `dropout` | 0.25 | 0.35 |
| `weight_decay` | 1e-4 | 0.01 |
| `label_smoothing` | 0.05 | 0.10 |
| `freeze_backbone_epochs` | 5 | 8 |
| `backbone_lr_multiplier` | 0.05 | 0.01 |
| `early_stopping.patience` | 15 | 6 |
| `unfreeze_encoder_layers` | 2 | 1 |
| M3_M1 `sampling.mode` | proportional | uniform |

## 日志 / TensorBoard

单槽覆盖目录（`replace_log_dir: true`）：

- `logs_sdavt_v3_r4/SDAVT_R4_M3_M1_roberta`
- `logs_sdavt_v3_r4/SDAVT_R4_M3_M7_combo`

## 重训结果 (2026-06-30)

| Job | Best F1@ep | Best Acc | ep8 解冻后 | Tier-2 |
|-----|------------|----------|------------|--------|
| M3_M1_roberta | **0.6823 @ ep17** | 0.6958 | 稳定，无崩溃 | PASS |
| **M3_M7_combo（新冠军）** | **0.6957 @ ep31** | **0.7121** | 稳定，无崩溃 | PASS |

ep8 解冻后 `collapse_detected: NO`；MELD 冠军已从 M3_M3_uniform 切换为 **M3_M7_combo**。

自动监控：`scripts/watch_p3_antioverfit_milestones.sh`（tmux: `watch_p3_milestones`）
日志：`outputs_sdavt_v3_r4/status/p3_antioverfit_milestone_watch.log`
