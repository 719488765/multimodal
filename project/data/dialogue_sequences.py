"""对话级序列数据集与 collate。

旧训练循环以“单话轮”为样本（T=1），ESF 的状态编码器因此退化为单步。本模块把
同一对话的连续话轮组织为 ``(B, T, ...)`` 的窗口，使状态迁移与
``previous_emotions`` 真正有输入。

用法（train.py）::

    base = MultimodalDataset(root, split="train", config=cfg)
    ds = DialogueSequenceDataset(base, window=8, stride=4)
    loader = DataLoader(ds, batch_size=8, collate_fn=sequence_collate_fn)

对话键的解析顺序：
1. 样本自带 ``dialogue_id``；
2. MELD 的 ``sample_id``（``meld_{split}_{row}``）经 CSV 行号映射到 Dialogue_ID；
3. 都不可用时，按数据顺序切连续窗口（退化模式，日志会警告）。
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)

_TEXT_KEYS = {
    "text_input_ids",
    "text_attention_mask",
    "context_text_input_ids",
    "context_text_attention_mask",
}


def _pad_1d(items: List[torch.Tensor], pad_value: float = 0.0) -> torch.Tensor:
    return pad_sequence(items, batch_first=True, padding_value=pad_value)


def _pad_text(items: List[torch.Tensor]) -> torch.Tensor:
    """文本序列按长度补齐。

    dataset 产出的 ``text_input_ids`` 可能是 ``(L,)`` 也可能是 ``(1, L)``；后者直接
    送 ``pad_sequence`` 会因为第 1 维长度不同而报错，这里先压掉长度 1 的前导维。
    """
    squeezed = [
        v.squeeze(0) if (v.dim() == 2 and v.size(0) == 1) else v for v in items
    ]
    return pad_sequence(squeezed, batch_first=True, padding_value=0)


def _stack_step_field(key: str, steps: List[Dict[str, Any]]) -> Optional[torch.Tensor]:
    vals = [s[key] for s in steps if s.get(key) is not None]
    if not vals:
        return None
    if len(vals) != len(steps):  # 该窗口内有样本缺这个字段：补零
        ref = vals[0]
        vals = [
            s[key] if s.get(key) is not None else torch.zeros_like(ref) for s in steps
        ]
    first = vals[0]
    if not torch.is_tensor(first):
        return None
    if key in _TEXT_KEYS or first.dim() == 1:
        return _pad_text(vals)
    if first.dim() == 2:
        # 预提取特征 (d,) / (L,d) 或音频波形 (L,)
        if all(v.shape == first.shape for v in vals):
            return torch.stack(vals, dim=0)
        return pad_sequence([v.reshape(v.size(0), -1) for v in vals], batch_first=True)
    if first.dim() == 3:  # (C,H,W)
        return torch.stack(vals, dim=0)
    if first.dim() == 4:  # (T,C,H,W) 单话轮多帧
        return torch.stack(vals, dim=0)
    return None


def sequence_collate_fn(batch: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """batch: list of windows; each window is a list of per-utterance sample dicts."""
    B = len(batch)
    T = max(len(w) for w in batch)
    lengths = torch.tensor([len(w) for w in batch], dtype=torch.long)
    sequence_mask = torch.zeros(B, T, dtype=torch.float32)
    for b, w in enumerate(batch):
        sequence_mask[b, : len(w)] = 1.0

    out: Dict[str, Any] = {"sequence_mask": sequence_mask, "lengths": lengths}
    keys = set()
    for w in batch:
        for s in w:
            keys.update(s.keys())
    for key in keys:
        if key in {"sample_id", "dialogue_id"}:
            out[key] = [[s.get(key) for s in w] for w in batch]
            continue
        if key == "emotion_label":
            labels = torch.full((B, T), -1, dtype=torch.long)
            for b, w in enumerate(batch):
                for t, s in enumerate(w):
                    labels[b, t] = int(s["emotion_label"])
            out["emotion_label"] = labels
            continue
        if key == "emotion_dimensions":
            dims = torch.zeros(B, T, 2, dtype=torch.float32)
            for b, w in enumerate(batch):
                for t, s in enumerate(w):
                    dims[b, t] = torch.as_tensor(s["emotion_dimensions"]).float()
            out["emotion_dimensions"] = dims
            continue
        if key == "dataset_id":
            ds = torch.full((B, T), -1, dtype=torch.long)
            for b, w in enumerate(batch):
                for t, s in enumerate(w):
                    ds[b, t] = int(s.get("dataset_id", -1))
            out["dataset_id"] = ds
            continue
        # 逐时间步特征
        padded_steps: List[List[Dict[str, Any]]] = []
        for w in batch:
            steps = list(w) + [dict() for _ in range(T - len(w))]
            padded_steps.append(steps)
        per_t = []
        for t in range(T):
            step_samples = [w[t] for w in padded_steps]
            v = _stack_step_field(key, step_samples)
            if v is not None:
                per_t.append(v)
        if per_t:
            # 每项形状 (B, ...) -> (B, T, ...)
            out[key] = torch.stack(per_t, dim=1)
    return out


def _metadata_entries(base: Dataset) -> Optional[List[dict]]:
    """低成本取出每个样本的元数据字典（含 sample_id），不触发媒体解码。

    `MultimodalDataset.data_list` 已经是 ``[{'sample_id': ..., 'video_path': ...}, ...]``，
    直接读它可以把窗口构建从「逐样本 __getitem__」降到纯字符串处理；
    若外层套了 ``Subset`` 则按 indices 取子集。
    `"""
    subset = getattr(base, "indices", None)
    holder = getattr(base, "dataset", base)
    entries = getattr(holder, "data_list", None)
    if not entries:
        return None
    if subset is not None:
        try:
            return [entries[i] for i in subset]
        except (IndexError, TypeError):
            return None
    return list(entries)


class DialogueSequenceDataset(Dataset):
    """把逐话轮数据集包装成对话窗口数据集。"""

    def __init__(
        self,
        base: Dataset,
        window: int = 8,
        stride: int = 4,
        dialogue_key: Optional[Callable[[Dict[str, Any], int], Optional[str]]] = None,
        max_windows: int = 0,
    ) -> None:
        if window < 2:
            raise ValueError("window must be >= 2 to exercise temporal state modeling")
        self.base = base
        self.window = int(window)
        self.stride = max(1, int(stride))
        self._windows: List[List[int]] = []

        entries = _metadata_entries(base)
        if entries is not None:
            keys: List[Optional[str]] = []
            for i, item in enumerate(entries):
                key = None
                if dialogue_key is not None:
                    key = dialogue_key(item, i)
                if key is None:
                    key = item.get("dialogue_id")
                keys.append(str(key) if key is not None else None)
            logger.info(
                "DialogueSequenceDataset: metadata path over %d samples (no media decode)",
                len(entries),
            )
        else:
            keys = []
            for i in range(len(base)):
                sample = base[i]
                key = None
                if dialogue_key is not None:
                    key = dialogue_key(sample, i)
                if key is None:
                    key = sample.get("dialogue_id") if isinstance(sample, dict) else None
                keys.append(str(key) if key is not None else None)

        if all(k is not None for k in keys):
            groups: Dict[str, List[int]] = {}
            for i, k in enumerate(keys):
                groups.setdefault(k, []).append(i)
            for idxs in groups.values():
                for start in range(0, max(len(idxs) - self.window + 1, 1), self.stride):
                    chunk = idxs[start : start + self.window]
                    if len(chunk) >= 2:
                        self._windows.append(chunk)
            logger.info(
                "DialogueSequenceDataset: %d dialogues -> %d windows (window=%d)",
                len(groups),
                len(self._windows),
                self.window,
            )
        else:
            logger.warning(
                "DialogueSequenceDataset: no dialogue key on %d/%d samples; "
                "falling back to consecutive windows (not true dialogues)",
                sum(k is None for k in keys),
                len(keys),
            )
            for start in range(0, max(len(base) - self.window + 1, 1), self.stride):
                self._windows.append(list(range(start, min(start + self.window, len(base)))))
        if max_windows and len(self._windows) > max_windows:
            self._windows = self._windows[:max_windows]

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int):
        return [self.base[i] for i in self._windows[idx]]

    def __getattr__(self, name: str):
        # 委托 data_list 等属性给底层数据集（避免递归）。
        if name in {"base", "_windows", "window", "stride"}:
            raise AttributeError(name)
        return getattr(self.base, name)


def build_meld_dialogue_key(config: dict, split: str) -> Optional[Callable[[Dict[str, Any], int], Optional[str]]]:
    """从 MELD CSV 构建 sample_id -> Dialogue_ID 的映射函数；CSV 缺失时返回 None。"""
    try:
        from data.meld_context import resolve_meld_csv, parse_meld_sample_id
    except Exception:  # pragma: no cover
        return None
    csv_path = resolve_meld_csv(split, config)
    if csv_path is None:
        return None
    import csv as _csv

    row_to_dialogue: Dict[int, str] = {}
    with open(csv_path, encoding="utf-8") as f:
        for row_idx, row in enumerate(_csv.DictReader(f), start=1):
            row_to_dialogue[row_idx] = str(row.get("Dialogue_ID", row_idx))
    logger.info("MELD dialogue map: %s -> %d rows", csv_path.name, len(row_to_dialogue))

    def _key(sample: Dict[str, Any], idx: int) -> Optional[str]:
        parsed = parse_meld_sample_id(str(sample.get("sample_id", "")))
        if parsed is None:
            return None
        _split, row_idx = parsed
        return row_to_dialogue.get(row_idx)

    return _key
