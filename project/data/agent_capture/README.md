# Agent 采集闭环数据

用于浏览器录制场景微调，目录结构（与 `MultimodalDataset` 一致）：

```
agent_capture/
  train/
    video/   meld_train_00001.mp4 或 .webm
    audio/   meld_train_00001.wav
    text/    meld_train_00001.txt   # 可选台词
    labels/  meld_train_00001.txt   # 第一行: happy|sad|neutral|...
  val/
    ...
```

标签第一行使用标准类名：`happy sad angry fear neutral anxious other`

建议每类至少 15 条，总计 ≥50 条。录制后运行：

```bash
cd project
python3 scripts/organize_agent_capture.py --src /path/to/raw_recordings
```
