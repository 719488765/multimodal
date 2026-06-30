#!/usr/bin/env python3
# Author: AI
# Date: 2026-03-31
# Description: 汇总 logs_rerun 各实验的 last/best 指标到 CSV 和 Markdown

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional


def to_float(val: str) -> Optional[float]:
    if val is None:
        return None
    val = val.strip()
    if val == "":
        return None
    try:
        return float(val)
    except ValueError:
        return None


def pick_best(rows: List[Dict], key: str) -> Dict:
    valid = [r for r in rows if r.get(key) is not None]
    if not valid:
        return {"epoch": None, key: None}
    return max(valid, key=lambda r: r[key])


def pick_min(rows: List[Dict], key: str) -> Dict:
    valid = [r for r in rows if r.get(key) is not None]
    if not valid:
        return {"epoch": None, key: None}
    return min(valid, key=lambda r: r[key])


def summarize_run(metrics_csv: Path) -> Optional[Dict]:
    run_dir = metrics_csv.parent.name
    val_rows: List[Dict] = []
    train_rows: List[Dict] = []

    with metrics_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            phase = row.get("phase", "")
            record = {
                "epoch": int(row["epoch"]),
                "loss": to_float(row.get("loss", "")),
                "accuracy": to_float(row.get("accuracy", "")),
                "precision": to_float(row.get("precision", "")),
                "recall": to_float(row.get("recall", "")),
                "f1": to_float(row.get("f1", "")),
            }
            if phase == "val":
                val_rows.append(record)
            elif phase == "train":
                train_rows.append(record)

    if not val_rows:
        return None

    val_rows.sort(key=lambda x: x["epoch"])
    train_rows.sort(key=lambda x: x["epoch"])

    last_val = val_rows[-1]
    best_acc = pick_best(val_rows, "accuracy")
    best_f1 = pick_best(val_rows, "f1")
    min_val_loss = pick_min(val_rows, "loss")

    train_first = train_rows[0] if train_rows else None
    train_last = train_rows[-1] if train_rows else None

    return {
        "run": run_dir,
        "val_rows": len(val_rows),
        "val_epoch_start": val_rows[0]["epoch"],
        "val_epoch_end": val_rows[-1]["epoch"],
        "last_val_epoch": last_val["epoch"],
        "last_val_loss": last_val["loss"],
        "last_acc": last_val["accuracy"],
        "last_precision": last_val["precision"],
        "last_recall": last_val["recall"],
        "last_f1": last_val["f1"],
        "best_acc_epoch": best_acc.get("epoch"),
        "best_acc": best_acc.get("accuracy"),
        "best_acc_f1": best_acc.get("f1"),
        "best_f1_epoch": best_f1.get("epoch"),
        "best_f1": best_f1.get("f1"),
        "best_f1_acc": best_f1.get("accuracy"),
        "min_val_loss_epoch": min_val_loss.get("epoch"),
        "min_val_loss": min_val_loss.get("loss"),
        "train_epoch_start": train_first["epoch"] if train_first else None,
        "train_epoch_end": train_last["epoch"] if train_last else None,
        "train_loss_start": train_first["loss"] if train_first else None,
        "train_loss_end": train_last["loss"] if train_last else None,
    }


def fmt(v: Optional[float], nd: int = 6) -> str:
    if v is None:
        return ""
    return f"{v:.{nd}f}"


def write_csv(records: List[Dict], output_csv: Path) -> None:
    fields = [
        "run",
        "val_rows",
        "val_epoch_start",
        "val_epoch_end",
        "last_val_epoch",
        "last_val_loss",
        "last_acc",
        "last_precision",
        "last_recall",
        "last_f1",
        "best_acc_epoch",
        "best_acc",
        "best_acc_f1",
        "best_f1_epoch",
        "best_f1",
        "best_f1_acc",
        "min_val_loss_epoch",
        "min_val_loss",
        "train_epoch_start",
        "train_epoch_end",
        "train_loss_start",
        "train_loss_end",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow(r)


def write_markdown(records: List[Dict], output_md: Path) -> None:
    lines = []
    lines.append("# Rerun Results Summary")
    lines.append("")
    lines.append("| run | val_range | last(acc/f1/loss) | best_acc(epoch,val) | best_f1(epoch,val) | min_val_loss(epoch,val) |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for r in records:
        val_range = f"{r['val_epoch_start']}..{r['val_epoch_end']} ({r['val_rows']})"
        last = f"{fmt(r['last_acc'])}/{fmt(r['last_f1'])}/{fmt(r['last_val_loss'])}"
        best_acc = f"{r['best_acc_epoch']},{fmt(r['best_acc'])}"
        best_f1 = f"{r['best_f1_epoch']},{fmt(r['best_f1'])}"
        min_loss = f"{r['min_val_loss_epoch']},{fmt(r['min_val_loss'])}"
        lines.append(f"| `{r['run']}` | {val_range} | {last} | {best_acc} | {best_f1} | {min_loss} |")
    lines.append("")
    output_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总 logs_rerun 实验结果")
    parser.add_argument("--logs_dir", default="logs_rerun", help="日志目录，默认 logs_rerun")
    parser.add_argument(
        "--output_csv",
        default="outputs_rerun/rerun_results_summary.csv",
        help="输出 CSV 路径",
    )
    parser.add_argument(
        "--output_md",
        default="outputs_rerun/rerun_results_summary.md",
        help="输出 Markdown 路径",
    )
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    metrics_files = sorted(logs_dir.glob("*/metrics.csv"))
    records = []
    for m in metrics_files:
        s = summarize_run(m)
        if s is not None:
            records.append(s)

    records.sort(key=lambda x: x["run"])
    write_csv(records, output_csv)
    write_markdown(records, output_md)

    print(f"summary runs: {len(records)}")
    print(f"csv: {output_csv}")
    print(f"md : {output_md}")


if __name__ == "__main__":
    main()
