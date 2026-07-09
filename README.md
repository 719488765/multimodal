# multimodal — 多模态情感分析（离线训练 + 在线智能体）

毕业论文实验仓库，包含：

- **`project/`** — 三数据集（CREMA-D / MELD / CMU-MOSEI）训练、R4 实验队列、checkpoint 与 TensorBoard  
- **`emotion-agent/`** — FastAPI + React 在线演示：A/V 采集、ASR、情绪推理、LLM 对话  

## 入口文档

| 文档 | 说明 |
|------|------|
| [**全栈技术架构与实验总结**](project/docs/MASTER_SYSTEM_ARCHITECTURE_AND_EXPERIMENT_SUMMARY.md) | **推荐首读**：SDAVT 模型架构 + R4 主结果 + emotion-agent 全栈（v2 架构重构，无文档索引充数） |
| [系统架构总览](project/docs/SYSTEM_ARCHITECTURE_OVERVIEW.md) | 精简版架构 |
| [R4 实验结果](project/docs/SDAVT_V3_R4_EXPERIMENT_RESULTS.md) | 55 job 指标表 |
| [在线 Agent 架构](emotion-agent/docs/ARCHITECTURE.md) | 8000/9010/11434 数据流 |
| [数据准备](project/use_data.md) | 数据集下载与整理 |

## 快速启动 Agent（默认 M3_M7 R4 冠军）

```bash
bash project/scripts/apply_deploy_preset.sh sdavt_meld_v3_r4
cd emotion-agent && FORCE_RESTART=1 ./scripts/start_demo.sh
# 浏览器: http://127.0.0.1:8000
```

TensorBoard（R4）：`http://127.0.0.1:6008`（logdir: `project/logs_sdavt_v3_r4/`）
