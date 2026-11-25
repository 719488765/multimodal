"""E5：汇总三 seed 结果，给出 mean±std。

输入：logs_e5/E5_<arm>_seed<seed>_meld_<timestamp>/metrics.csv
输出：reports/e5_seed_summary.csv 与 .md

口径：
- 每个 run 取"最优验证 Macro-F1"及其对应轮的 Accuracy；
- 跨 seed 报告 mean±std（样本标准差，ddof=1）；
- seed 42 的结果来自 logs_e2 的对应 run，一并纳入。
"""
from __future__ import annotations

import re
import statistics
from pathlib import Path

import pandas as pd

ROOT = Path("/mnt/sda1/lizhichun_24/code/multimodal/project")
OUT = ROOT / "reports"
OUT.mkdir(parents=True, exist_ok=True)

# arm -> (E5 目录前缀, seed42 的 logs_e2 目录名前缀)
ARMS = {
    "F1_standard": ("E5_F1_standard_seed", "E2_F1_standard_meld_20261124_033122"),
    "F2_lf_text": ("E5_F2_lf_text_seed", "E2_F2_lf_text_meld_20261124_064712"),
    "F4_es_noseq": ("E5_F4_es_noseq_seed", "E2_F4_es_noseq_meld_20261123_235014"),
}


def best_of(csv_path: Path):
    raw = csv_path.read_bytes()
    text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8-sig")
    rows = [r for r in __import__("csv").DictReader(text.splitlines())
            if r.get("phase") == "val" and r.get("f1")]
    if not rows:
        return None
    top = max(rows, key=lambda r: float(r["f1"]))
    return float(top["f1"]), float(top["accuracy"]), int(top["epoch"])


def find_run(prefix: str, base: Path) -> Path | None:
    cands = sorted(p for p in base.glob(f"{prefix}*") if p.is_dir())
    return cands[-1] if cands else None


def main() -> int:
    summary = []
    for arm, (p5, p42) in ARMS.items():
        per_seed = {}
        d42 = find_run(p42, ROOT / "logs_e2")
        if d42 and (d42 / "metrics.csv").is_file():
            per_seed[42] = best_of(d42 / "metrics.csv")
        for seed in (1337, 2024):
            d = find_run(f"{p5}{seed}", ROOT / "logs_e5")
            if d and (d / "metrics.csv").is_file():
                per_seed[seed] = best_of(d / "metrics.csv")
        if not per_seed:
            continue
        f1s = [v[0] for v in per_seed.values()]
        accs = [v[1] for v in per_seed.values()]
        summary.append({
            "arm": arm,
            "seeds": "/".join(str(s) for s in sorted(per_seed)),
            "n": len(f1s),
            "f1_mean": round(statistics.mean(f1s), 4),
            "f1_std": round(statistics.stdev(f1s), 4) if len(f1s) > 1 else 0.0,
            "acc_mean": round(statistics.mean(accs), 4),
            "acc_std": round(statistics.stdev(accs), 4) if len(accs) > 1 else 0.0,
            "detail": "; ".join(f"{s}:{v[0]:.4f}" for s, v in sorted(per_seed.items())),
        })

    if not summary:
        print("还没有可汇总的 E5 结果")
        return 1

    df = pd.DataFrame(summary)
    df.to_csv(OUT / "e5_seed_summary.csv", index=False, encoding="utf-8-sig")

    lines = ["# E5 三种子汇总", ""]
    lines.append("| 臂 | 纳入种子 | n | Macro-F1 (mean±std) | Accuracy (mean±std) | 各 seed 最优 F1 |")
    lines.append("|---|---|---|---|---|---|")
    for r in summary:
        lines.append("| {arm} | {seeds} | {n} | {f1_mean:.4f} ± {f1_std:.4f} | "
                     "{acc_mean:.4f} ± {acc_std:.4f} | {detail} |".format(**r))
    lines += ["", "标准差为样本标准差（ddof=1）；n<3 时仅作参考，不做推断统计。"]
    (OUT / "e5_seed_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
