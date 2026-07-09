#!/usr/bin/env python3
"""Assemble MASTER document v2: architecture-focused narrative + auto R4 metrics only."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import master_doc_sections

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "docs" / "MASTER_SYSTEM_ARCHITECTURE_AND_EXPERIMENT_SUMMARY.md"
METRICS_SNIPPET = PROJECT_ROOT / "outputs_sdavt_v3_r4" / "status" / "master_doc_metrics_snippet.md"
CHAMP_SNIPPET = PROJECT_ROOT / "outputs_sdavt_v3_r4" / "status" / "master_doc_champions_snippet.md"
R4_REPORT = PROJECT_ROOT / "docs" / "SDAVT_V3_R4_EXPERIMENT_RESULTS.md"


def load(path: Path, default: str = "") -> str:
    return path.read_text(encoding="utf-8") if path.is_file() else default


def extract_r4_phase_tables(report: str) -> str:
    if "## p0_fix" in report:
        return report.split("## p0_fix", 1)[1].split("---", 1)[0].strip()
    return report


def main() -> int:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    champ = load(CHAMP_SNIPPET)
    metrics = load(METRICS_SNIPPET)
    r4_phases = extract_r4_phase_tables(load(R4_REPORT))

    body = master_doc_sections.build_all(r4_phases, champ, metrics)

    header = f"""# 多模态情感分析 — 全栈技术架构与实验总结（Master）

**版本**：2026-07-09 v2（R4 close-out）  
**状态**：R4 **close-out 完成**；MELD **M3_M7** F1=0.696；MOSEI **F_O_ES** F1=0.679；CREMA **C4_C3** Acc=0.605（Tier-2 CLOSE-OUT）；Agent 默认 **sdavt_meld_v3_r4**  
**配套图**：[`figures/system_architecture_figure.svg`](figures/system_architecture_figure.svg)  
**指标刷新**：`python scripts/build_master_doc_metrics.py` → `assemble_master_document.py`（{now}）

> 本文档以**技术架构**为主线：SDAVT 多模态情绪模型（编码器 + 融合 + 训练）与 emotion-agent 在线系统（React + FastAPI + ASR + 推理 + LLM）。  
> 实验部分聚焦 **R4 主轨**；AP0–AP4 仅作探索背景。不罗列仓库内数百篇 md 路径；运维脚本见附录 A 索引。

---

## 目录

### Part I 背景与方法
- [第1章 研究背景与相关工作](#第1章-研究背景与相关工作)
- [第2章 实验设计思路与演进](#第2章-实验设计思路与演进)

### Part II SDAVT 模型架构
- [第3章 总体架构 MultimodalEmotionModel](#第3章-总体架构-multimodalemotionmodel)
- [第4章 模态编码器](#第4章-模态编码器)
- [第5章 融合模块深读](#第5章-融合模块深读)
- [第6章 数据与训练流程](#第6章-数据与训练流程)

### Part III R4 实验结果
- [第7章 R4 主结果与分阶段解读](#第7章-r4-主结果与分阶段解读)
- [第8章 工程问题与修复要点](#第8章-工程问题与修复要点)

### Part IV Emotion Agent
- [第9章 智能体总体架构](#第9章-智能体总体架构)
- [第10章 前端模块与交互](#第10章-前端模块与交互)
- [第11章 后端模块与推理链路](#第11章-后端模块与推理链路)
- [第12章 中文域优化与部署](#第12章-中文域优化与部署)

### 附录
- [附录 A 核心脚本索引](#附录-a-核心脚本索引)
- [附录 B Checkpoint 与端口](#附录-b-checkpoint-与端口)
- [附录 C 论文数字锚点](#附录-c-论文数字锚点)

---

# Part I 背景与方法

"""

    footer = """

---

## 修订记录

| 日期 | 说明 |
|------|------|
| 2026-07-07 | v1 分批扩写（已废弃大量索引充数） |
| 2026-07-07 | **v2 架构重构**：章节重排；内化文献/模型/Agent 架构；R4 聚焦；删除 302 md 列表与逐脚本 bash |
| 2026-07-09 | **R4 close-out**：CREMA champion C4_C3 Acc=0.605；MELD V 重训 FAIL；Agent 测试主轨 |
"""

    doc = header + body + footer
    OUT.write_text(doc, encoding="utf-8")
    lines = len(doc.splitlines())
    print(f"[OK] wrote {OUT} ({lines} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
