#!/usr/bin/env python3
# Author: AI
# Date: 2026-03-31
# Description: 重算关键 checkpoint 指标并回填论文主表（best_f1 / best_acc / last）

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Dict, Optional


def parse_run_config(run_name: str, config_dir: Path) -> Optional[Path]:
    r = run_name.lower()
    mapping = [
        ("avt", "emotion_shift", "config_AVT_noDA_emotion_shift.yaml"),
        ("avt", "da_w002", "config_AVT_DA_w002.yaml"),
        ("avt", "da_w005", "config_AVT_DA_w005.yaml"),
        ("avt", "da_w010", "config_AVT_DA_w010.yaml"),
        ("avt", "da", "config_AVT_DA.yaml"),
        ("avt", "noda", "config_AVT_noDA.yaml"),
        ("vt", "noda", "config_VT_noDA.yaml"),
        ("at", "da", "config_AT_DA.yaml"),
        ("at", "noda", "config_AT_noDA.yaml"),
        ("v_only", "", "config_video_only.yaml"),
        ("t_pretrain", "", "config_text_only.yaml"),
        ("a_pretrain", "", "config_audio_only.yaml"),
    ]

    for must1, must2, cfg in mapping:
        if must1 in r and (must2 == "" or must2 in r):
            p = config_dir / cfg
            if p.exists():
                return p
    return None


def call_recompute(recompute_script: Path, config_path: Path, checkpoint_path: Path, batch_size: int) -> Optional[Dict]:
    cmd = [
        "python3",
        str(recompute_script),
        "--config",
        str(config_path),
        "--checkpoint",
        str(checkpoint_path),
        "--split",
        "val",
        "--batch_size",
        str(batch_size),
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError:
        return None

    # 兼容前后可能有 warning 文本，尝试截取最后一个 JSON 对象
    out = res.stdout.strip()
    if not out:
        return None
    start = out.rfind("{")
    end = out.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(out[start : end + 1])
    except json.JSONDecodeError:
        return None


def read_summary_map(summary_csv: Path) -> Dict[str, Dict]:
    m: Dict[str, Dict] = {}
    if not summary_csv.exists():
        return m
    with summary_csv.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m[row["run"]] = row
    return m


def to_int(v: str) -> Optional[int]:
    try:
        return int(v)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="重算并回填论文主表")
    parser.add_argument("--paper_table_csv", default="outputs_rerun/paper_table_main.csv")
    parser.add_argument("--summary_csv", default="outputs_rerun/rerun_results_summary.csv")
    parser.add_argument("--config_dir", default="config/rerun")
    parser.add_argument("--checkpoints_dir", default="checkpoints_rerun")
    parser.add_argument("--recompute_script", default="scripts/recompute_val_metrics.py")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--output_csv", default="outputs_rerun/paper_table_main_recomputed.csv")
    parser.add_argument("--output_md", default="outputs_rerun/paper_table_main_recomputed.md")
    args = parser.parse_args()

    paper_table_csv = Path(args.paper_table_csv)
    summary_csv = Path(args.summary_csv)
    config_dir = Path(args.config_dir)
    checkpoints_dir = Path(args.checkpoints_dir)
    recompute_script = Path(args.recompute_script)
    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if not paper_table_csv.exists():
        print(f"[ERROR] paper table not found: {paper_table_csv}")
        return
    if not summary_csv.exists():
        print(f"[ERROR] summary csv not found: {summary_csv}")
        return

    summary_map = read_summary_map(summary_csv)
    rows = []
    with paper_table_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    out_rows = []
    for r in rows:
        run = r.get("run", "")
        summary = summary_map.get(run)
        config_path = parse_run_config(run, config_dir)

        # 基础字段拷贝
        row = dict(r)
        row["recompute_config"] = str(config_path) if config_path else ""
        row["recompute_status"] = "skip"
        row["best_f1_recalc_acc"] = ""
        row["best_f1_recalc_f1"] = ""
        row["best_acc_recalc_acc"] = ""
        row["best_acc_recalc_f1"] = ""
        row["last_recalc_acc"] = ""
        row["last_recalc_f1"] = ""

        if not summary or not config_path or not config_path.exists():
            out_rows.append(row)
            continue

        bf1_ep = to_int(summary.get("best_f1_epoch", ""))
        bacc_ep = to_int(summary.get("best_acc_epoch", ""))
        last_ep = to_int(summary.get("last_val_epoch", ""))

        ckpt_dir = checkpoints_dir / run
        if not ckpt_dir.exists():
            out_rows.append(row)
            continue

        ok_any = False
        for tag, ep in [("best_f1", bf1_ep), ("best_acc", bacc_ep), ("last", last_ep)]:
            if ep is None:
                continue
            ckpt = ckpt_dir / f"checkpoint_pretrain_epoch_{ep}.pth"
            if not ckpt.exists():
                continue
            met = call_recompute(recompute_script, config_path, ckpt, args.batch_size)
            if not met:
                continue
            ok_any = True
            row[f"{tag}_recalc_acc"] = f"{met.get('accuracy', 0.0):.6f}"
            row[f"{tag}_recalc_f1"] = f"{met.get('f1', 0.0):.6f}"

        row["recompute_status"] = "ok" if ok_any else "partial_or_missing"
        out_rows.append(row)

    # 写 CSV
    fieldnames = list(out_rows[0].keys()) if out_rows else []
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    # 写 Markdown
    lines = []
    lines.append("# Paper Main Table (Recomputed)")
    lines.append("")
    lines.append("| run | modality | DA | fusion | status | best_f1(recalc acc/f1) | best_acc(recalc acc/f1) | last(recalc acc/f1) |")
    lines.append("|---|---|---|---|---|---:|---:|---:|")
    for r in out_rows:
        bf1 = f"{r.get('best_f1_recalc_acc','')}/{r.get('best_f1_recalc_f1','')}"
        bacc = f"{r.get('best_acc_recalc_acc','')}/{r.get('best_acc_recalc_f1','')}"
        last = f"{r.get('last_recalc_acc','')}/{r.get('last_recalc_f1','')}"
        lines.append(
            f"| `{r.get('run','')}` | {r.get('modality','')} | {r.get('da','')} | {r.get('fusion','')} | {r.get('recompute_status','')} | {bf1} | {bacc} | {last} |"
        )
    output_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"input rows: {len(rows)}")
    print(f"output rows: {len(out_rows)}")
    print(f"csv: {output_csv}")
    print(f"md : {output_md}")


if __name__ == "__main__":
    main()
