#!/usr/bin/env python3
# Author: AI
# Date: 2026-03-31
# Description: 从 rerun 汇总结果生成论文主表初稿（CSV + Markdown）

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def parse_run_name(run: str) -> Dict[str, str]:
    """
    解析 run 名称，提取模态/DA/融合/训练范式字段。
    适配 RERUN_* 命名与常见历史命名。
    """
    r = run.lower()

    # 默认值
    modality = "unknown"
    da = "unknown"
    fusion = "standard"
    training_regime = "pretrain"
    variant = ""

    # 模态识别（按优先级）
    if "avt" in r:
        modality = "AVT"
    elif "vt" in r:
        modality = "VT"
    elif "at" in r:
        modality = "AT"
    elif "v_only" in r:
        modality = "V-only"
    elif "_t_" in f"_{r}_" or "t_pretrain" in r or "text_only" in r:
        modality = "T-only"
    elif "_a_" in f"_{r}_" or "a_pretrain" in r or "audio_only" in r:
        modality = "A-only"

    # DA
    if "noda" in r:
        da = "noDA"
    elif "_da" in r or r.endswith("da"):
        da = "DA"

    # 融合策略
    if "emotion_shift" in r or "_es" in r:
        fusion = "emotion_shift"
    elif "leader" in r:
        fusion = "leader_follower"
    elif "two_stage" in r:
        fusion = "two_stage"

    # 训练范式（当前主线多为 pretrain）
    if "finetune" in r:
        training_regime = "finetune"
    elif "pretrain" in r:
        training_regime = "pretrain"

    # 变体（DA权重网格等）
    if "_w002" in r:
        variant = "da_w002"
    elif "_w005" in r:
        variant = "da_w005"
    elif "_w010" in r:
        variant = "da_w010"

    return {
        "modality": modality,
        "da": da,
        "fusion": fusion,
        "training_regime": training_regime,
        "variant": variant,
    }


def to_float(v: str):
    try:
        return float(v)
    except Exception:
        return None


def fmt(v, nd=6):
    if v is None or v == "":
        return ""
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return str(v)


def read_summary_csv(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_main_table(rows: List[Dict]) -> List[Dict]:
    out = []
    for r in rows:
        run = r.get("run", "")
        meta = parse_run_name(run)
        out.append(
            {
                "run": run,
                "modality": meta["modality"],
                "fusion": meta["fusion"],
                "da": meta["da"],
                "training_regime": meta["training_regime"],
                "variant": meta["variant"],
                "val_range": f"{r.get('val_epoch_start','')}..{r.get('val_epoch_end','')}",
                "last_acc": fmt(r.get("last_acc", "")),
                "last_f1": fmt(r.get("last_f1", "")),
                "best_acc_epoch": r.get("best_acc_epoch", ""),
                "best_acc": fmt(r.get("best_acc", "")),
                "best_f1_epoch": r.get("best_f1_epoch", ""),
                "best_f1": fmt(r.get("best_f1", "")),
                "min_val_loss_epoch": r.get("min_val_loss_epoch", ""),
                "min_val_loss": fmt(r.get("min_val_loss", "")),
            }
        )

    # 排序：主线优先 + 其余按 run
    order = {
        "AT": 1,
        "T-only": 2,
        "A-only": 3,
        "V-only": 4,
        "VT": 5,
        "AVT": 6,
        "unknown": 99,
    }
    out.sort(key=lambda x: (order.get(x["modality"], 99), x["da"], x["fusion"], x["run"]))
    return out


def write_csv(rows: List[Dict], path: Path):
    fields = [
        "run",
        "modality",
        "fusion",
        "da",
        "training_regime",
        "variant",
        "val_range",
        "last_acc",
        "last_f1",
        "best_acc_epoch",
        "best_acc",
        "best_f1_epoch",
        "best_f1",
        "min_val_loss_epoch",
        "min_val_loss",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_md(rows: List[Dict], path: Path):
    lines = []
    lines.append("# Paper Main Table Draft")
    lines.append("")
    lines.append("| run | modality | fusion | DA | regime | last(acc/f1) | best_acc(epoch,val) | best_f1(epoch,val) |")
    lines.append("|---|---|---|---|---|---:|---:|---:|")
    for r in rows:
        last = f"{r['last_acc']}/{r['last_f1']}"
        bacc = f"{r['best_acc_epoch']},{r['best_acc']}"
        bf1 = f"{r['best_f1_epoch']},{r['best_f1']}"
        lines.append(
            f"| `{r['run']}` | {r['modality']} | {r['fusion']} | {r['da']} | {r['training_regime']} | {last} | {bacc} | {bf1} |"
        )
    lines.append("")
    lines.append("> 注：precision/recall/f1 如存在历史口径差异，建议以 `recompute_val_metrics.py` 复核后再定稿。")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="构建论文主表初稿")
    parser.add_argument(
        "--summary_csv",
        default="outputs_rerun/rerun_results_summary.csv",
        help="输入汇总CSV路径",
    )
    parser.add_argument(
        "--output_csv",
        default="outputs_rerun/paper_table_main.csv",
        help="输出主表CSV路径",
    )
    parser.add_argument(
        "--output_md",
        default="outputs_rerun/paper_table_main.md",
        help="输出主表Markdown路径",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary_csv)
    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    rows = read_summary_csv(summary_path)
    table = build_main_table(rows)
    write_csv(table, output_csv)
    write_md(table, output_md)

    print(f"input rows: {len(rows)}")
    print(f"table rows: {len(table)}")
    print(f"csv: {output_csv}")
    print(f"md : {output_md}")


if __name__ == "__main__":
    main()
