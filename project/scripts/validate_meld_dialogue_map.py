"""校验 MELD sample_id -> Dialogue_ID 映射是否与整理后的数据一致。

背景：`data/meld_context.py` 与 `data/dialogue_sequences.py` 都假设
`meld_{split}_{idx:04d}` 中的 idx 等于官方 CSV 的 1-based 行号（organize_meld.py 的约定）。
本脚本验证三件事：

1. CSV 行数与数据集中的 `meld_{split}_*.txt` 文件数是否一致；
2. 相邻 sample_id 是否落在同一 Dialogue_ID（对话在 CSV 中连续排列）；
3. 反例的具体样本，便于判断是否需要重建索引。

用法：
    python scripts/validate_meld_dialogue_map.py --project-root .
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple

CSV_TEMPLATE = "downloads/MELD/data/MELD/{split}_sent_emo.csv"
SPLIT_ALIAS = {"train": "train", "val": "dev", "test": "test"}
SAMPLE_RE = re.compile(r"^meld_(train|val|test)_(\d+)\.txt$")


def load_rows(csv_path: Path) -> List[dict]:
    with csv_path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def scan_split_files(project_root: Path, split: str) -> Dict[int, Path]:
    text_dir = project_root / "data" / split / "text"
    found: Dict[int, Path] = {}
    if not text_dir.is_dir():
        return found
    for p in text_dir.iterdir():
        m = SAMPLE_RE.match(p.name)
        if m and m.group(1) == split:
            found[int(m.group(2))] = p
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    args = ap.parse_args()
    root = Path(args.project_root).resolve()

    ok_all = True
    for split in ("train", "val", "test"):
        csv_path = root / CSV_TEMPLATE.format(split=SPLIT_ALIAS[split])
        print(f"\n=== split={split} ===")
        if not csv_path.is_file():
            print(f"  [MISS] {csv_path}")
            ok_all = False
            continue

        rows = load_rows(csv_path)
        dialogue_of_row: Dict[int, str] = {}
        for i, row in enumerate(rows, start=1):
            dialogue_of_row[i] = str(row.get("Dialogue_ID", ""))

        uniq = len(set(dialogue_of_row.values()))
        print(f"  CSV 行数={len(rows)}  对话数={uniq}  -> {csv_path.name}")

        files = scan_split_files(root, split)
        print(f"  数据集样本数={len(files)}")
        if files:
            missing = [i for i in range(1, len(rows) + 1) if i not in files]
            print(f"  未出现在数据集中的行号数={len(missing)}"
                  + (f"（前 5：{missing[:5]}）" if missing else ""))
            if len(files) != len(rows):
                print("  [WARN] 样本数与 CSV 行数不一致，需确认整理时是否有过滤")
        else:
            print("  [WARN] 未找到数据文件，跳过顺序校验")
            continue

        # 顺序校验：每个 dialogue 的行号应当是连续区间
        boundaries = 0
        examples: List[Tuple[int, str, str]] = []
        prev_dlg = None
        for idx in sorted(files):
            dlg = dialogue_of_row.get(idx)
            if dlg is None:
                continue
            if prev_dlg is not None and dlg != prev_dlg:
                boundaries += 1
            prev_dlg = dlg

        # 真正要检测的是“同一 dialogue 的行号是否被打散”
        pos: Dict[str, List[int]] = {}
        for idx in sorted(files):
            dlg = dialogue_of_row.get(idx)
            if dlg is not None:
                pos.setdefault(dlg, []).append(idx)
        non_contiguous = 0
        for dlg, idxs in pos.items():
            if idxs and (idxs[-1] - idxs[0] + 1) != len(idxs):
                non_contiguous += 1
                if len(examples) < 3:
                    examples.append((idxs[0], dlg, f"跨度 {idxs[0]}..{idxs[-1]} 但只有 {len(idxs)} 行"))
        print(f"  对话切换次数={boundaries}  非连续对话数={non_contiguous}")
        if examples:
            for a, b, c in examples:
                print(f"    例：首行 {a} Dialogue_ID={b} {c}")

        # 说明：DialogueSequenceDataset 按 dialogue_key 分组取样，窗口不会跨对话拼样，
        # 因此行号是否连续只影响“能否当作顺序索引”，不构成失败条件。
        # 真正的失败条件是映射缺失（无 CSV）或样本行号越界。
        out_of_range = [i for i in files if i > len(rows)]
        if out_of_range:
            ok_all = False
            print(f"  [FAIL] 有 {len(out_of_range)} 个样本行号超出 CSV（前 5：{out_of_range[:5]}）")
        elif non_contiguous:
            print("  [NOTE] 存在被拆散的对话（通常是少数样本缺媒体），"
                  "分组取样不受影响；若这些对话要计入评估，需补齐样本")
        else:
            print("  [OK] 映射可用，对话内样本完整且连续")

    print("\n结论：" + ("全部通过" if ok_all else "存在需要处理的问题"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
