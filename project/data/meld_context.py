"""MELD dialogue context index for context-lite (k previous utterances)."""
from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MELD_CSV_BY_SPLIT: Dict[str, List[Path]] = {
    "train": [
        PROJECT_ROOT / "downloads" / "MELD" / "data" / "MELD" / "train_sent_emo.csv",
        PROJECT_ROOT / "downloads" / "MELD" / "videos" / "train_sent_emo.csv",
    ],
    "val": [
        PROJECT_ROOT / "downloads" / "MELD" / "data" / "MELD" / "dev_sent_emo.csv",
    ],
    "test": [
        PROJECT_ROOT / "downloads" / "MELD" / "data" / "MELD" / "test_sent_emo.csv",
    ],
}
MELD_SAMPLE_RE = re.compile(r"^meld_(train|val|test)_(\d{4})$", re.IGNORECASE)


def resolve_meld_csv(split: str, config: Optional[dict] = None) -> Optional[Path]:
    """Locate MELD annotation CSV for project split (train/val/test)."""
    data_cfg = (config or {}).get("data", {})
    custom = data_cfg.get("meld_csv_path")
    if custom:
        p = Path(custom)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.exists():
            return p

    per_split = data_cfg.get("meld_csv_by_split") or {}
    if split in per_split:
        p = Path(per_split[split])
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.exists():
            return p

    for cand in MELD_CSV_BY_SPLIT.get(split, []):
        if cand.exists():
            return cand
    return None


def _load_meld_rows(csv_path: Path) -> List[dict]:
    rows: List[dict] = []
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def build_meld_context_map(
    split: str, context_window: int, config: Optional[dict] = None
) -> Dict[str, List[str]]:
    """
    Map meld_{split}_{idx:04d} -> up to k previous utterance texts in the same dialogue.
    Uses CSV row order (1-based idx) consistent with organize_meld.py.
    """
    if context_window <= 0:
        return {}

    csv_path = resolve_meld_csv(split, config)
    if csv_path is None:
        logger.warning(
            "MELD context_window=%d but CSV not found for split=%s",
            context_window,
            split,
        )
        return {}

    rows = _load_meld_rows(csv_path)
    dialogues: Dict[str, List[tuple]] = {}
    for csv_idx, row in enumerate(rows, start=1):
        dialogue_id = str(row.get("Dialogue_ID", ""))
        try:
            utterance_id = int(row.get("Utterance_ID", 0))
        except (TypeError, ValueError):
            utterance_id = csv_idx
        utterance = (row.get("Utterance") or "").strip()
        dialogues.setdefault(dialogue_id, []).append((csv_idx, utterance_id, utterance))

    for utterances in dialogues.values():
        utterances.sort(key=lambda x: x[1])

    context_map: Dict[str, List[str]] = {}
    for utterances in dialogues.values():
        for pos, (csv_idx, _uid, _text) in enumerate(utterances):
            prev_texts: List[str] = []
            start = max(0, pos - context_window)
            for j in range(start, pos):
                prev_texts.append(utterances[j][2])
            while len(prev_texts) < context_window:
                prev_texts.insert(0, "")
            sample_id = f"meld_{split}_{csv_idx:04d}"
            context_map[sample_id] = prev_texts[-context_window:]

    logger.info(
        "MELD context index: split=%s window=%d csv=%s samples=%d",
        split,
        context_window,
        csv_path.name,
        len(context_map),
    )
    return context_map


def parse_meld_sample_id(sample_id: str) -> Optional[tuple]:
    m = MELD_SAMPLE_RE.match(sample_id or "")
    if not m:
        return None
    return m.group(1).lower(), int(m.group(2))
