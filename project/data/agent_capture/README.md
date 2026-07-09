# Agent 采集闭环数据（中文场景微调）

用于浏览器录制场景微调，纠正 webcam + **中文 ASR** 域偏移。

## 目录结构

```
agent_capture/
  train/
    video/   sample_001.webm
    audio/   sample_001.wav
    text/    sample_001.txt   # 可选：ASR 或手输中文台词
    labels/  sample_001.txt   # 第一行: happy|sad|neutral|...
  val/
    ...
```

标签第一行使用标准类名：`happy sad angry fear neutral anxious other`

## 采集建议（中文优化方案）

| 项 | 建议 |
|----|------|
| 数量 | **≥100 条**（happy/sad/neutral/angry 各 ≥20） |
| 时长 | 每条 **≥5 秒** 中文语音 |
| 内容 | 明确情感词（「很开心」「很难过」）+ 自然对话 |
| 格式 | wav 16kHz mono + webm/mp4 或 4 帧截图 |

## 整理与微调

```bash
cd project

# 1. 整理原始录制
python3 scripts/organize_agent_capture.py --src /path/to/raw_recordings

# 2. 一键准备 + 可选微调（需 GPU）
bash scripts/prepare_agent_capture_finetune.sh /path/to/raw_recordings --finetune

# 3. 微调完成后切换 preset
bash scripts/apply_deploy_preset.sh sdavt_meld_zh_agent
cd ../emotion-agent && FORCE_RESTART=1 ./scripts/start_demo.sh
```

训练配置：[`config/sdavt_v3_r4/p3_m3/meld/M3_M7_chinese_agent.yaml`](../config/sdavt_v3_r4/p3_m3/meld/M3_M7_chinese_agent.yaml)

Resume 自 R4 冠军 M3_M7 V/A/fusion，仅替换 **bert-base-chinese** text 分支。

## 中文基准回归（无需 GPU）

```bash
python3 scripts/eval_zh_agent_benchmark.py
python3 scripts/eval_agent_capture_cases.py
```

基准用例：[`data/agent_benchmark/zh_cases.json`](../data/agent_benchmark/zh_cases.json)

## 在线工程优化（已默认启用）

`config/config_agent_deploy.yaml` → `chinese_agent`：

- 中文 ASR → **bypass 英文 roberta text encoder**
- `leader_modal: audio`
- ASR 校准 + 多模态仲裁

详见 [`emotion-agent/docs/ARCHITECTURE.md`](../../emotion-agent/docs/ARCHITECTURE.md) §0。
