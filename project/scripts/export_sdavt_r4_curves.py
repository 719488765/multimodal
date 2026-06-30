#!/usr/bin/env python3
"""从 R4 metrics.csv 导出论文曲线图（四宫格：Acc/F1/cls_ce/train loss）。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_metrics(metrics_csv: Path) -> Dict[str, List]:
    train_loss, val_acc, val_f1, val_ce = [], [], [], []
    with metrics_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ep = int(row["epoch"])
            phase = row.get("phase", "")
            if phase == "train":
                train_loss.append((ep, float(row.get("loss") or 0)))
            elif phase == "val":
                val_acc.append((ep, float(row.get("accuracy") or 0)))
                val_f1.append((ep, float(row.get("f1") or 0)))
                ce = row.get("cls_ce_unweighted", "")
                if ce:
                    val_ce.append((ep, float(ce)))
    return {
        "train_loss": train_loss,
        "val_acc": val_acc,
        "val_f1": val_f1,
        "val_ce": val_ce,
    }


def plot_run(metrics_csv: Path, out_png: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed; skip figure export")
        return False

    data = load_metrics(metrics_csv)
    if not data["val_acc"]:
        return False

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(metrics_csv.parent.name, fontsize=10)

    if data["val_acc"]:
        xs, ys = zip(*data["val_acc"])
        axes[0, 0].plot(xs, ys, "b-o", markersize=3)
    axes[0, 0].set_title("Val Accuracy")
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].grid(True, alpha=0.3)

    if data["val_f1"]:
        xs, ys = zip(*data["val_f1"])
        axes[0, 1].plot(xs, ys, "g-o", markersize=3)
    axes[0, 1].set_title("Val macro-F1")
    axes[0, 1].set_xlabel("epoch")
    axes[0, 1].grid(True, alpha=0.3)

    if data["val_ce"]:
        xs, ys = zip(*data["val_ce"])
        axes[1, 0].plot(xs, ys, "r-o", markersize=3)
    axes[1, 0].set_title("Val cls_ce_unweighted")
    axes[1, 0].set_xlabel("epoch")
    axes[1, 0].grid(True, alpha=0.3)

    if data["train_loss"]:
        xs, ys = zip(*data["train_loss"])
        axes[1, 1].plot(xs, ys, "k-o", markersize=3)
    axes[1, 1].set_title("Train loss_total")
    axes[1, 1].set_xlabel("epoch")
    axes[1, 1].grid(True, alpha=0.3)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="logs_sdavt_v3_r4")
    parser.add_argument("--output-dir", default="outputs_sdavt_v3_r4/figures")
    parser.add_argument("--run", default="", help="Single run dir name (optional)")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    out_dir = Path(args.output_dir)
    pattern = f"{args.run}/metrics.csv" if args.run else "*/metrics.csv"
    count = 0
    for metrics in sorted(log_dir.glob(pattern)):
        out_png = out_dir / f"{metrics.parent.name}_curves.png"
        if plot_run(metrics, out_png):
            count += 1
            print(f"[OK] {out_png}")
    print(f"Exported {count} figure(s) -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
