# MASTER 文档进度

**目标文档**：[`MASTER_SYSTEM_ARCHITECTURE_AND_EXPERIMENT_SUMMARY.md`](MASTER_SYSTEM_ARCHITECTURE_AND_EXPERIMENT_SUMMARY.md)

## v2 架构重构（2026-07-07）

| 项 | 状态 |
|----|------|
| 章节重排 Part I–IV | done |
| 删除 302 md 列表 / 逐脚本 bash / 55 job 卡片 | done |
| CFN-ESA / 编码器 / Agent 架构扩写 | done |
| AP 压缩为第 2 章探索摘要 | done |
| R4 phase 表 + champion 自动表保留 | done |
| 附录精简为 A/B/C | done |

**原则**：以技术架构干货为主，行数约 **900–1200**（去充数后）；指标仍由脚本自动刷新。

## 刷新命令

```bash
cd project && source scripts/r4_env.sh
python scripts/build_master_doc_metrics.py
python scripts/assemble_master_document.py
```

## 产出脚本

| 文件 | 说明 |
|------|------|
| `scripts/master_doc_sections.py` | 叙事章节内容（v2） |
| `scripts/assemble_master_document.py` | 组装 MASTER |
| `scripts/build_master_doc_metrics.py` | R4 指标片段 |
