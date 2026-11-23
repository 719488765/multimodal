"""探针：验证 MELD 对话级序列管线（数据集 -> collate -> 张量形状）。

不做训练，只证明三件事：
1. `DialogueSequenceDataset` 能在真实 MELD 数据上按 Dialogue_ID 切出窗口；
2. `sequence_collate_fn` 产出的形状是 (B, T, ...) 而不是 (B, ...)；
3. 用右移标签构造 previous_emotions 的形状与 ESF 期望一致。

用法：
    python scripts/probe_meld_sequence_pipeline.py --config config/xxx.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import MultimodalDataset  # noqa: E402
from data.dialogue_sequences import (  # noqa: E402
    DialogueSequenceDataset,
    build_meld_dialogue_key,
    sequence_collate_fn,
)
from scripts.train import _subset_by_datasets  # noqa: E402


def describe(name: str, value) -> None:
    if torch.is_tensor(value):
        print(f"  {name:28s} tensor {tuple(value.shape)} {value.dtype}")
    elif isinstance(value, list):
        print(f"  {name:28s} list[{len(value)}] 首元素类型={type(value[0]).__name__}")
    else:
        print(f"  {name:28s} {type(value).__name__}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", default="meld", help="与 train.py --dataset 同义")
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    cfg.setdefault("training", {}).setdefault("finetune", {})["datasets"] = [args.dataset]
    base = MultimodalDataset(cfg["data"]["root_dir"], split="val", config=cfg)
    base = _subset_by_datasets(base, cfg, "finetune")
    print(f"基础数据集(val, {args.dataset}) 样本数 = {len(base)}")

    key_fn = build_meld_dialogue_key(cfg, "val")
    print(f"对话键函数 = {'已构建' if key_fn else '缺失（将退化为顺序窗口）'}")

    ds = DialogueSequenceDataset(base, window=args.window, stride=args.stride,
                                 dialogue_key=key_fn, max_windows=args.batch * 4)
    print(f"窗口数 = {len(ds)}  (window={args.window} stride={args.stride})")

    batch = [ds[i] for i in range(min(args.batch, len(ds)))]
    print(f"\n第一个 batch 的窗口长度 = {[len(w) for w in batch]}")
    out = sequence_collate_fn(batch)
    print("collate 输出：")
    for k in sorted(out):
        describe(k, out[k])

    ok = True
    if out.get("sequence_mask") is None:
        print("[FAIL] 缺少 sequence_mask")
        ok = False
    else:
        b, t = out["sequence_mask"].shape
        if t < 2:
            print(f"[FAIL] T={t}，状态建模没有输入")
            ok = False
        else:
            print(f"[OK] T={t} > 1，状态编码器有真实时间步")

    for key in ("text_input_ids", "audio", "video"):
        v = out.get(key)
        if v is None:
            continue
        if v.dim() >= 3 and v.shape[0] == out["sequence_mask"].shape[0] and v.shape[1] == out["sequence_mask"].shape[1]:
            print(f"[OK] {key} 形状 {tuple(v.shape)} 带 T 维")
        else:
            print(f"[WARN] {key} 形状 {tuple(v.shape)} 与 (B,T) 不匹配，检查 collate")

    # previous_emotions 构造（与 train.py 训练分支一致）
    labels = out["emotion_label"]
    k = int(cfg["model"]["output"]["emotion_classes"])
    eye = torch.nn.functional.one_hot(labels.clamp(min=0), num_classes=k).float()
    uniform = torch.full((labels.shape[0], 1, k), 1.0 / k)
    prev = torch.cat([uniform, eye[:, :-1]], dim=1)
    print(f"\nprevious_emotions 形状 = {tuple(prev.shape)}  (期望 (B,T,{k}))")
    if tuple(prev.shape) != (labels.shape[0], labels.shape[1], k):
        print("[FAIL] previous_emotions 形状不符")
        ok = False

    print("\n结论：" + ("管线可用" if ok else "管线有问题"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
