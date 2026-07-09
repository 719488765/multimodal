#!/usr/bin/env python3
"""Generate comprehensive R4 experiment report (markdown) in project/docs/."""

from __future__ import annotations

import csv
import json
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "outputs_sdavt_v3_r4" / "experiment_queue.json"
LOG_DIR = ROOT / "logs_sdavt_v3_r4"
CKPT_DIR = ROOT / "checkpoints_sdavt_v3_r4"
AUDIT = ROOT / "outputs_sdavt_v3_r4" / "tables" / "r4_training_audit.json"
FUSION_WINNERS = ROOT / "outputs_sdavt_v3_r4" / "tables" / "r4_fusion_winners.json"
CLOSEOUT = ROOT / "outputs_sdavt_v3_r4" / "status" / "r4_closeout_snapshot_20260709.json"
OUT = ROOT / "docs" / "R4_FULL_EXPERIMENT_REPORT.md"

PHASE_DESC = {
    "p0_fix": "P0 融合修复验证（非 ES 融合 sanity）",
    "p1_baseline": "P1 单域 AVT + emotion_shift 主基线",
    "p2_fusion": "P2 五融合策略对比（选定 ES 为后续主融合）",
    "p3_c3": "P3-C CREMA 配方消融（Acc 导向）",
    "p3_m3": "P3-M MELD 配方消融（F1 导向）",
    "p4_modal": "P4 模态消融（7 种模态组合 × 3 数据集）",
}

# Post-queue CREMA Tier-2 重训（不在 55-job 队列内）
EXTRA_JOBS = [
    {
        "id": "C4_C1_combo_acc",
        "phase": "p3_c_plus",
        "dataset": "crema",
        "run_dir": "SDAVT_R4_C4_C1_combo_acc",
        "config": "config/sdavt_v3_r4/p3_c_plus/crema/C4_C1_combo_acc.yaml",
        "note": "failed_nan",
    },
    {
        "id": "C4_C2_c3_base_acc",
        "phase": "p3_c_plus",
        "dataset": "crema",
        "run_dir": "SDAVT_R4_C4_C2_c3_base_acc",
        "config": "config/sdavt_v3_r4/p3_c_plus/crema/C4_C2_c3_base_acc.yaml",
    },
    {
        "id": "C4_C3_c3_warmstart_acc",
        "phase": "p3_c_plus",
        "dataset": "crema",
        "run_dir": "SDAVT_R4_C4_C3_c3_warmstart_acc",
        "config": "config/sdavt_v3_r4/p3_c_plus/crema/C4_C3_c3_warmstart_acc.yaml",
        "note": "close-out champion Acc=0.605",
    },
]

MODALITY_LETTERS = {"video": "V", "audio": "A", "text": "T", "physiological": "P"}


def load_yaml(rel: str) -> dict:
    p = ROOT / rel
    if not p.is_file():
        return {}
    with p.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def modality_str(cfg: dict) -> str:
    m = cfg.get("model", {}).get("modalities", {})
    parts = []
    for key, letter in [("use_text", "T"), ("use_audio", "A"), ("use_video", "V"), ("use_physiological", "P")]:
        if m.get(key):
            parts.append(letter)
    return "+".join(parts) if parts else "—"


def backbone_summary(cfg: dict) -> str:
    model = cfg.get("model", {})
    bits = []
    m = model.get("modalities", {})
    if m.get("use_text"):
        bits.append(f"T:{model.get('text', {}).get('backbone', '?')}")
    if m.get("use_audio"):
        bits.append(f"A:{model.get('audio', {}).get('backbone', 'wav2vec').split('/')[-1]}")
    if m.get("use_video"):
        v = model.get("video", {})
        bits.append(f"V:{v.get('backbone', '?')}" + (f"/npy" if v.get("input_type") == "npy" else ""))
    attn = model.get("attention", {})
    bits.append(f"fusion={attn.get('fusion_strategy', '?')}")
    if attn.get("leader_modal"):
        bits.append(f"leader={attn['leader_modal']}")
    return "; ".join(bits)


def read_metrics(run_dir: str) -> Dict[str, Any]:
    csv_path = LOG_DIR / run_dir / "metrics.csv"
    out: Dict[str, Any] = {"run_dir": run_dir, "has_metrics": False}
    if not csv_path.is_file():
        return out
    val_rows = []
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("phase") != "val":
                continue
            try:
                val_rows.append(
                    {
                        "epoch": int(row["epoch"]),
                        "f1": float(row["f1"]) if row.get("f1") else None,
                        "acc": float(row["accuracy"]) if row.get("accuracy") else None,
                        "loss": float(row["loss"]) if row.get("loss") else None,
                    }
                )
            except (ValueError, KeyError):
                continue
    if not val_rows:
        return out
    best_f1 = max(val_rows, key=lambda r: r["f1"] or 0)
    best_acc = max(val_rows, key=lambda r: r["acc"] or 0)
    out.update(
        {
            "has_metrics": True,
            "epochs_val": len(val_rows),
            "best_f1": best_f1["f1"],
            "best_f1_ep": best_f1["epoch"],
            "best_acc": best_acc["acc"],
            "best_acc_ep": best_acc["epoch"],
            "last_f1": val_rows[-1]["f1"],
            "last_acc": val_rows[-1]["acc"],
        }
    )
    return out


def find_checkpoints(run_dir: str) -> List[str]:
    d = CKPT_DIR / run_dir
    if not d.is_dir():
        # timestamped dirs
        matches = sorted(CKPT_DIR.glob(f"{run_dir}*"))
        if matches:
            d = matches[-1]
        else:
            return []
    return sorted(p.name for p in d.glob("*.pth")) if d.is_dir() else []


def collapse_flag(job: dict, metrics: dict) -> str:
    bf1 = job.get("best_val_f1") or metrics.get("best_f1")
    if bf1 is None:
        return "—"
    ds = job.get("dataset", "")
    thr = 0.06 if ds == "crema" else 0.10
    ep0_f1 = job.get("ep0_val_f1")
    if ep0_f1 is not None and bf1 <= ep0_f1 + 0.01:
        return "collapse?"
    if bf1 < thr:
        return "✗ low"
    return "✓"


def hostname_ip() -> str:
    try:
        return socket.gethostbyname(socket.gethostname())
    except OSError:
        return "127.0.0.1"


def build_job_row(job: dict) -> Dict[str, Any]:
    cfg_path = job.get("config", "")
    cfg = load_yaml(cfg_path)
    run_dir = job.get("run_dir") or f"SDAVT_R4_{job.get('id', '')}"
    metrics = read_metrics(run_dir)
    ckpts = find_checkpoints(run_dir)
    return {
        **job,
        "modalities": modality_str(cfg),
        "backbones": backbone_summary(cfg),
        "metrics": metrics,
        "checkpoints": ckpts,
        "ckpt_dir": str(CKPT_DIR / run_dir),
        "log_dir": str(LOG_DIR / run_dir),
        "collapse": collapse_flag(job, metrics),
    }


def fmt_metric(v: Optional[float], ep: Optional[int] = None) -> str:
    if v is None:
        return "—"
    s = f"{v:.4f}"
    if ep is not None:
        s += f" @ ep{ep}"
    return s


def main() -> None:
    queue = json.loads(QUEUE.read_text(encoding="utf-8"))
    jobs = list(queue.get("jobs", []))
    for extra in EXTRA_JOBS:
        m = read_metrics(extra["run_dir"])
        jobs.append(
            {
                **extra,
                "status": "done",
                "best_val_f1": m.get("best_f1"),
                "best_val_acc": m.get("best_acc"),
                "best_val_f1_ep": m.get("best_f1_ep"),
                "best_val_acc_ep": m.get("best_acc_ep"),
            }
        )

    rows = [build_job_row(j) for j in jobs]
    by_phase: Dict[str, List[dict]] = {}
    for r in rows:
        by_phase.setdefault(r.get("phase", "?"), []).append(r)

    audit = json.loads(AUDIT.read_text(encoding="utf-8")) if AUDIT.is_file() else {}
    closeout = json.loads(CLOSEOUT.read_text(encoding="utf-8")) if CLOSEOUT.is_file() else {}
    fusion = json.loads(FUSION_WINNERS.read_text(encoding="utf-8")) if FUSION_WINNERS.is_file() else {}

    log_runs = sorted(p.name for p in LOG_DIR.iterdir() if p.is_dir()) if LOG_DIR.is_dir() else []
    ckpt_runs = sorted(p.name for p in CKPT_DIR.iterdir() if p.is_dir()) if CKPT_DIR.is_dir() else []

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    host = hostname_ip()
    tb_url = f"http://{host}:6008"
    tb_local = "http://127.0.0.1:6008"

    lines: List[str] = []
    w = lines.append

    w("# SDAVT v3 R4 完整实验报告")
    w("")
    w(f"*生成时间：{ts}*  ")
    w(f"*生成脚本：`scripts/build_r4_full_experiment_report.py`*  ")
    w(f"*数据源：`experiment_queue.json`（55 jobs）+ P3-C+ 重训（3 jobs）+ `logs_sdavt_v3_r4/` metrics*")
    w("")
    w("---")
    w("")
    w("## 目录")
    w("")
    w("1. [实验总览与 Tier-2 结论](#1-实验总览与-tier-2-结论)")
    w("2. [实验过程与时间线](#2-实验过程与时间线)")
    w("3. [实验环境与资产索引](#3-实验环境与资产索引)")
    w("4. [TensorBoard 访问](#4-tensorboard-访问)")
    w("5. [分阶段实验设计与结果](#5-分阶段实验设计与结果)")
    w("6. [融合策略消融（P2）](#6-融合策略消融p2)")
    w("7. [配方消融（P3）](#7-配方消融p3)")
    w("8. [模态消融（P4）](#8-模态消融p4)")
    w("9. [Close-out 并行重训](#9-close-out-并行重训)")
    w("10. [结果分析与论文叙事](#10-结果分析与论文叙事)")
    w("11. [Checkpoint 与日志完整清单](#11-checkpoint-与日志完整清单)")
    w("")
    w("---")
    w("")
    w("## 1. 实验总览与 Tier-2 结论")
    w("")
    w("R4（Round 4）为 SDAVT v3 **论文主轨**实验线，共 **6 个阶段 + Close-out 重训**：")
    w("")
    w("| 阶段 | 目的 | Job 数 |")
    w("|------|------|--------|")
    w("| P0 | 融合模块修复验证 | 6 |")
    w("| P1 | 三数据集 ES 主基线 | 3 |")
    w("| P2 | 五融合策略对比 | 14 |")
    w("| P3-C | CREMA Acc 配方消融 | 3 (+3 close-out) |")
    w("| P3-M | MELD F1 配方消融 | 8 |")
    w("| P4 | 模态组合消融 Table 4 | 21 |")
    w("| **合计（队列）** | | **55 done** |")
    w("")
    w("### Tier-2 验收（论文口径，2026-07-09 close-out）")
    w("")
    w("| 数据集 | 指标目标 | **最终 Champion** | Best F1 | Best Acc | 判定 |")
    w("|--------|----------|-------------------|---------|----------|------|")
    w("| **MELD** | F1≥0.59, Acc≥0.62 | **M3_M7_combo** | **0.696** @ ep31 | **0.712** @ ep31 | **PASS** |")
    w("| **MOSEI** | F1≥0.67 | **F_O_ES**（P2 融合冠军） | **0.679** @ ep12 | 0.727 @ ep12 | **PASS** |")
    w("| **CREMA** | Acc≥0.63 | **C4_C3** warm-start | 0.606 @ ep65 | **0.605** @ ep65 | **CLOSE-OUT**（差 2.5pp） |")
    w("")
    w("**Audit：** P0=0，collapse（严格审计）=0，tier2_fail=1（CREMA 未达 0.63，已由 C4_C3 close-out 定论）。")
    w("")
    w("---")
    w("")
    w("## 2. 实验过程与时间线")
    w("")
    w("| 时间段 | 阶段 | 关键事件 |")
    w("|--------|------|----------|")
    w("| 2026-06-22 ~ 06-23 | P0 | 融合修复 sanity（STD/TS/LFT 等） |")
    w("| 2026-06-23 | P1 | 三数据集 ES 主基线 R4_B_* 完成 |")
    w("| 2026-06-23 ~ 06-24 | P2 | 14 组融合对比；**ES 锁定**为 P3/P4 默认 |")
    w("| 2026-06-25 ~ 06-26 | P3 | MELD 8 组配方 + CREMA 3 组配方并行 |")
    w("| 2026-06-26 ~ 06-30 | P4 | 21 组模态消融（Table 4） |")
    w("| 2026-07-07 ~ 07-09 | Close-out | C4_C3 warm-start + R4_A_M_V 重训；R4 GPU 线关闭 |")
    w("")
    w("**执行方式：** 双 GPU 队列 `sdavt_r4_worker.sh` + tmux 监控；指标自动写入 `metrics.csv` / TensorBoard。")
    w("")
    w("### 2.1 统一训练设置（R4 主轨典型值）")
    w("")
    w("| 项目 | 设置 |")
    w("|------|------|")
    w("| 模型 | `MultimodalEmotionModel` + YAML 配置驱动 |")
    w("| 优化器 | AdamW，lr=1e-4（finetune 可降），weight_decay=0.01 |")
    w("| 调度 | cosine + warmup；gradient_clip=1.0 |")
    w("| Batch | MELD/CREMA 常 batch=1 + grad_accum=2；MOSEI 视显存调整 |")
    w("| Early stopping | monitor=val_f1 或 val_acc，patience 6~10 |")
    w("| 视频 | ResNet50，112×112，4 frames，3.0s 窗口 |")
    w("| 音频 | wav2vec2-base（P1/P2/P4）或 **large**（P3 冠军配方） |")
    w("| 文本 | bert-base-uncased（P1/P4）或 **roberta-base**（M3_M7） |")
    w("| MOSEI 视频 | OpenFace2 npy 713-d 时序特征（`input_type: npy`） |")
    w("| 融合 | P2 后固定 **emotion_shift**，leader 默认 text（P4 单模态 yaml 内改 leader_modal） |")
    w("| 隔离目录 | `checkpoints_sdavt_v3_r4/`、`logs_sdavt_v3_r4/`、`outputs_sdavt_v3_r4/` |")
    w("")
    w("---")
    w("")
    w("## 3. 实验环境与资产索引")
    w("")
    w("| 资源 | 路径 | 数量 |")
    w("|------|------|------|")
    w(f"| 训练日志 | `logs_sdavt_v3_r4/` | **{len(log_runs)}** runs |")
    w(f"| Checkpoint | `checkpoints_sdavt_v3_r4/` | **{len(ckpt_runs)}** dirs |")
    w("| 队列状态 | `outputs_sdavt_v3_r4/experiment_queue.json` | 55 jobs |")
    w("| 指标表 | `outputs_sdavt_v3_r4/tables/r4_training_audit.json` | 63 runs w/ metrics |")
    w("| Close-out 快照 | `outputs_sdavt_v3_r4/status/r4_closeout_snapshot_20260709.json` | — |")
    w("")
    w("**统一模型骨架：** `MultimodalEmotionModel`（ResNet50 / Wav2Vec2 / BERT-RoBERTa + 可切换融合）  ")
    w("**默认融合（P2 后）：** `emotion_shift`（CFN-ESA 风格 Emotion-Shift + cross-attn）  ")
    w("**训练入口：** `scripts/train.py`；**队列 worker：** `scripts/sdavt_r4_worker.sh`")
    w("")
    w("---")
    w("")
    w("## 4. TensorBoard 访问")
    w("")
    w("所有 R4 训练曲线（loss / F1 / Acc / 分 loss 项）写入各 run 的 TensorBoard event 文件，logdir 统一为：")
    w("")
    w("```text")
    w("project/logs_sdavt_v3_r4/")
    w("```")
    w("")
    w("| 访问方式 | URL / 命令 |")
    w("|----------|------------|")
    w(f"| **本机浏览器** | [{tb_local}]({tb_local}) |")
    w(f"| **局域网 / SSH 转发** | `{tb_url}`（host={host}） |")
    w("| 启动命令 | `bash scripts/tensorboard_sdavt_r4.sh 6008` |")
    w("| tmux 会话 | `tmux attach -t sdavt_r4_tensorboard` |")
    w("")
    w("**查看全部训练图像：**")
    w("1. 浏览器打开 TensorBoard → 左侧 **Scalars** 可筛选 `train/`、`val/` 指标")
    w("2. 使用右上角 **Filter runs** 搜索 job id，如 `M3_M7`、`C4_C3`、`R4_A_M_V`")
    w("3. 每个 run 目录名 = 下表 `Run Dir` 列，与 TB 中 tag 前缀一致")
    w("")
    w("> 若通过 SSH 远程开发，请在 Cursor/VS Code **Ports** 面板转发 **6008**，浏览器访问 `http://127.0.0.1:6008`。")
    w("")
    w("---")
    w("")
    w("## 5. 分阶段实验设计与结果")
    w("")

    phase_order = ["p0_fix", "p1_baseline", "p2_fusion", "p3_c3", "p3_m3", "p4_modal", "p3_c_plus"]
    for phase in phase_order:
        batch = by_phase.get(phase, [])
        if not batch:
            continue
        desc = PHASE_DESC.get(phase, phase)
        w(f"### {phase} — {desc}")
        w("")
        w("| Job | Dataset | 模态 | 融合/骨干 | Best F1 | Best Acc | Ep | Collapse | Run Dir | CKPT |")
        w("|-----|---------|------|-----------|---------|----------|-----|----------|---------|------|")
        for r in sorted(batch, key=lambda x: (x.get("dataset", ""), x.get("id", ""))):
            m = r.get("metrics", {})
            bf1 = r.get("best_val_f1") or m.get("best_f1")
            bacc = r.get("best_val_acc") or m.get("best_acc")
            ep = r.get("best_val_f1_ep") or m.get("best_f1_ep")
            ck = "✓" if r.get("checkpoints") else "—"
            w(
                f"| {r.get('id')} | {r.get('dataset')} | {r.get('modalities')} | "
                f"{r.get('backbones', '')[:48]} | "
                f"{fmt_metric(bf1, ep)} | "
                f"{fmt_metric(bacc, r.get('best_val_acc_ep') or m.get('best_acc_ep'))} | "
                f"{r.get('epochs_done') or m.get('epochs_val', '—')} | "
                f"{r.get('collapse')} | `{r.get('run_dir', r.get('metrics', {}).get('run_dir', ''))}` | {ck} |"
            )
        w("")

    w("---")
    w("")
    w("## 6. 融合策略消融（P2）")
    w("")
    w("P2 在三数据集上对比 **5 种融合**：`standard` / `two_stage` / `leader_follower` / `functional_correlation` / `emotion_shift`。")
    w("**结论：** `emotion_shift`（ES）在 MELD/MOSEI 上稳定最优或并列最优，CREMA 上 ES 亦为 P2 冠军，故 P3/P4 全部固定 ES。")
    w("")
    w("| Dataset | P2 冠军 Job | Best F1 | Best Acc | Run |")
    w("|---------|-------------|---------|----------|-----|")
    for ds in ("crema", "meld", "mosei"):
        fw = fusion.get(ds, {})
        w(
            f"| {ds} | **{fw.get('job_id', '?')}** | "
            f"{fw.get('best_f1', 0):.4f} @ ep{fw.get('best_f1_ep')} | "
            f"{fw.get('best_acc', 0):.4f} | `{fw.get('run', '')}` |"
        )
    w("")
    w("**分融合明细（同数据集内对比）：**")
    w("")
    for ds in ("crema", "meld", "mosei"):
        w(f"#### {ds.upper()}")
        w("")
        w("| Fusion | Job | Best F1 | Best Acc |")
        w("|--------|-----|---------|----------|")
        for r in sorted(by_phase.get("p2_fusion", []), key=lambda x: x.get("best_val_f1") or 0, reverse=True):
            if r.get("dataset") != ds:
                continue
            w(
                f"| {r.get('fusion', '?')} | {r.get('id')} | "
                f"{fmt_metric(r.get('best_val_f1'), r.get('best_val_f1_ep'))} | "
                f"{fmt_metric(r.get('best_val_acc'), r.get('best_val_acc_ep'))} |"
            )
        w("")

    w("---")
    w("")
    w("## 7. 配方消融（P3）")
    w("")
    w("### 6.1 MELD（P3-M）— 目标 F1≥0.59")
    w("")
    w("| Job | 配方要点 | Best F1 | Best Acc | 相对 M0 |")
    w("|-----|----------|---------|----------|---------|")
    m0_f1 = next((r.get("best_val_f1") for r in by_phase.get("p3_m3", []) if r.get("id") == "M3_M0_baseline"), 0.608)
    for r in sorted(by_phase.get("p3_m3", []), key=lambda x: x.get("best_val_f1") or 0, reverse=True):
        bf1 = r.get("best_val_f1") or 0
        bacc = r.get("best_val_acc") or 0
        delta = bf1 - m0_f1
        note = r.get("id", "").replace("M3_", "")
        w(
            f"| **{r.get('id')}** | {note} | "
            f"**{bf1:.4f}** @ ep{r.get('best_val_f1_ep')} | "
            f"{bacc:.4f} | {delta:+.4f} |"
        )
    w("")
    w("**M3_M7_combo 配方（冠军）：** roberta-base + wav2vec2-large + ResNet50；`use_context_window=true`；`modality_dropout=0.1`；focal loss + label smoothing；dropout 0.35。")
    w("")
    w("### 6.2 CREMA（P3-C + P3-C+）— 目标 Acc≥0.63")
    w("")
    w("| Job | 配方要点 | Best Acc | Best F1 | 判定 |")
    w("|-----|----------|----------|---------|------|")
    crema_jobs = by_phase.get("p3_c3", []) + by_phase.get("p3_c_plus", [])
    for r in sorted(crema_jobs, key=lambda x: x.get("best_val_acc") or x.get("metrics", {}).get("best_acc") or 0, reverse=True):
        acc = r.get("best_val_acc") or r.get("metrics", {}).get("best_acc")
        f1 = r.get("best_val_f1") or r.get("metrics", {}).get("best_f1")
        verdict = r.get("note", "队列完成")
        if r.get("id") == "C4_C3_c3_warmstart_acc":
            verdict = "**CLOSE-OUT champion**"
        elif r.get("id") == "C4_C1_combo_acc":
            verdict = "failed_nan"
        acc_s = f"**{acc:.4f}**" if acc is not None else "—"
        ep = r.get("best_val_acc_ep") or r.get("metrics", {}).get("best_acc_ep")
        f1_s = f"{f1:.4f}" if f1 is not None else "—"
        w(
            f"| {r.get('id')} | {r.get('phase')} | "
            f"{acc_s} @ ep{ep if ep is not None else '—'} | "
            f"{f1_s} | {verdict} |"
        )
    w("")

    w("---")
    w("")
    w("## 8. 模态消融（P4）")
    w("")
    w("P4 在 ES 融合固定下，对 **7 种模态组合**（A/T/V/AT/AV/VT/AVT）× **3 数据集** 进行消融。")
    w("Job 命名：`R4_A_{C|M|O}_{modality}`（C=CREMA, M=MELD, O=MOSEI）。")
    w("")
    for ds, label in [("meld", "MELD"), ("mosei", "MOSEI"), ("crema", "CREMA")]:
        w(f"### {label}")
        w("")
        w("| 模态 | Job | Best F1 | Best Acc | Collapse | 分析 |")
        w("|------|-----|---------|----------|----------|------|")
        p4 = [r for r in by_phase.get("p4_modal", []) if r.get("dataset") == ds]
        avt_f1 = next((r.get("best_val_f1") for r in p4 if r.get("id", "").endswith("AVT")), None)
        for r in sorted(p4, key=lambda x: x.get("best_val_f1") or 0, reverse=True):
            mod = r.get("modalities", "?")
            bf1 = r.get("best_val_f1")
            analysis = ""
            if r.get("id", "").endswith("_V") and ds == "meld":
                analysis = "视频单模态极弱（对话情感依赖文本/音频）"
            elif r.get("id", "").endswith("_T") and bf1 and avt_f1:
                analysis = f"文本单模态接近全模态（{bf1:.3f} vs AVT {avt_f1:.3f}）"
            elif r.get("collapse", "").startswith("✗"):
                analysis = "低 F1 / collapse 标记"
            w(
                f"| {mod} | {r.get('id')} | "
                f"{fmt_metric(bf1, r.get('best_val_f1_ep'))} | "
                f"{fmt_metric(r.get('best_val_acc'), r.get('best_val_acc_ep'))} | "
                f"{r.get('collapse')} | {analysis} |"
            )
        w("")

    w("---")
    w("")
    w("## 9. Close-out 并行重训")
    w("")
    if closeout:
        pr = closeout.get("parallel_retrain", {})
        w("| Job | 完成时间 | Best 指标 | 判定 | Run |")
        w("|-----|----------|-----------|------|-----|")
        for key in ("C4_C3", "R4_A_M_V"):
            item = pr.get(key, {})
            w(
                f"| **{key}** | {item.get('completed_at', '—')} | "
                f"F1={item.get('best_f1', item.get('best_f1', '—'))} Acc={item.get('best_acc', '—')} | "
                f"{item.get('verdict', '—')} | `{item.get('run_dir', '')}` |"
            )
    w("")
    w("---")
    w("")
    w("## 10. 结果分析与论文叙事")
    w("")
    w("### 9.1 主要发现")
    w("")
    w("1. **融合：** Emotion-Shift 在三个数据集 P2 对比中均为首选，验证 CFN-ESA 风格 shift-aware 融合在本骨架上的有效性。")
    w("2. **MELD 配方：** M3_M7（RoBERTa-large 音频 + context window + modality dropout）相对 M0 提升 **+8.8pp F1**，为 Agent 默认 preset。")
    w("3. **MOSEI：** F_O_ES F1=0.679 达标；P4 全模态 R4_A_O_AVT F1=0.698 略高，但 P2 冠军仍为 ES 主轨配置。")
    w("4. **CREMA：** C3_C2 w2v-large Acc=0.567 → C4_C3 warm-start **0.605**，未达 Tier-2 0.63；激进改 recipe（C4_C1/C4_C2）退化。")
    w("5. **模态消融：** MELD 上 **T ≈ AVT**（F1 0.674 vs 0.682），**V-only F1≈0.27** 为任务固有难度，非训练失效。")
    w("")
    w("### 9.2 论文推荐数字")
    w("")
    w("| Table | 内容 | 推荐数值 |")
    w("|-------|------|----------|")
    w("| 主结果 MELD | M3_M7_combo | F1=**0.696**, Acc=**0.712** |")
    w("| 主结果 MOSEI | F_O_ES | F1=**0.679**, Acc=0.727 |")
    w("| 主结果 CREMA | C4_C3 | Acc=**0.605**, F1=0.606 |")
    w("| Table 4 脚注 | R4_A_M_V | F1≈0.269（V-only 下限） |")
    w("")
    w("### 9.3 Agent 部署映射")
    w("")
    w("| Preset | Checkpoint | 来源 Job |")
    w("|--------|------------|----------|")
    w("| `sdavt_meld_v3_r4` | `SDAVT_R4_M3_M7_combo/checkpoint_pretrain_best_f1.pth` | M3_M7_combo |")
    w("| `sdavt_mosei_r4` | `SDAVT_R4_F_O_ES_*/checkpoint_pretrain_best_f1.pth` | F_O_ES |")
    w("| `sdavt_crema_r4` | C3_C2 或 C4_C3 | 实验/close-out |")
    w("")
    w("---")
    w("")
    w("## 11. Checkpoint 与日志完整清单")
    w("")
    w(f"共 **{len(log_runs)}** 个 log run、**{len(ckpt_runs)}** 个 checkpoint 目录（含时间戳后缀变体）。")
    w("")
    w("| Run Dir | metrics.csv | Checkpoint (.pth) | TB Scalars |")
    w("|---------|-------------|-------------------|------------|")
    all_run_dirs = sorted(set(log_runs) | set(r.get("run_dir", "") for r in rows))
    for rd in all_run_dirs:
        if not rd:
            continue
        has_m = "✓" if (LOG_DIR / rd / "metrics.csv").is_file() else "—"
        ckpts = find_checkpoints(rd)
        ck = ", ".join(ckpts[:2]) + ("…" if len(ckpts) > 2 else "") if ckpts else "—"
        tb = "✓" if any((LOG_DIR / rd).glob("events.out.tfevents.*")) else "—"
        w(f"| `{rd}` | {has_m} | {ck} | {tb} |")
    w("")
    w("---")
    w("")
    w("*刷新：`python scripts/build_r4_full_experiment_report.py`*  ")
    w("*配套实时表：`docs/SDAVT_V3_R4_EXPERIMENT_RESULTS.md`（`build_sdavt_r4_report.py`）*")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {OUT} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
