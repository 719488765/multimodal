"""E2 测速：真实数据上测「取批 + 前向 + 反向」耗时，推算单 epoch。

不训练、不落盘，只用来决定 E2 的轮数与 batch 规模。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torch.utils.data import DataLoader  # noqa: E402

from data.dataset import MultimodalDataset  # noqa: E402
from data.dialogue_sequences import (  # noqa: E402
    DialogueSequenceDataset,
    build_meld_dialogue_key,
    sequence_collate_fn,
)
from models.multimodal_model import MultimodalEmotionModel  # noqa: E402
from scripts.train import _subset_by_datasets  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=5)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    dataset = (cfg.get("training", {}).get("finetune", {}) or {}).get("datasets", ["meld"])[0]
    temporal = cfg.get("data", {}).get("temporal", {})
    window = int(temporal.get("window", 8))
    stride = int(temporal.get("stride", 4))
    batch = int(temporal.get("batch_size", 2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(0),
              "| 已用 %.1f GB / 总 %.1f GB" % (
                  torch.cuda.memory_allocated(0) / 1e9,
                  torch.cuda.get_device_properties(0).total_memory / 1e9))

    t0 = time.time()
    base = MultimodalDataset(cfg["data"]["root_dir"], split="train", config=cfg)
    base = _subset_by_datasets(base, cfg, "finetune")
    ds = DialogueSequenceDataset(base, window=window, stride=stride,
                                 dialogue_key=build_meld_dialogue_key(cfg, "train"))
    print(f"train windows = {len(ds)}  ({dataset}, window={window}, stride={stride}, batch={batch})")
    print(f"数据集构建耗时 {time.time() - t0:.1f}s")

    loader = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0,
                        collate_fn=sequence_collate_fn)

    model = MultimodalEmotionModel(cfg).to(device)
    model.train()
    n_param = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"模型参数 {n_param:.1f}M")

    it = iter(loader)
    data_time = 0.0
    step_time = 0.0
    n = 0
    for i in range(args.steps):
        ta = time.time()
        try:
            b = next(it)
        except StopIteration:
            break
        data_time += time.time() - ta

        inputs = {
            "text_input_ids": b["text_input_ids"].to(device),
            "text_attention_mask": b["text_attention_mask"].to(device),
            "audio": b["audio"].to(device),
            "video": b["video"].to(device),
            "sequence_mask": b["sequence_mask"].to(device),
        }
        labels = b["emotion_label"].to(device)
        mask = b["sequence_mask"].to(device)

        tb = time.time()
        out = model(**inputs)
        logits = out["emotion_logits"] if isinstance(out, dict) else out
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none"
        )
        w = mask.reshape(-1)
        loss = ((loss * w).sum() / w.sum().clamp(min=1.0))
        loss.backward()
        model.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        step_time += time.time() - tb
        n += 1

    if n == 0:
        print("没有取到 batch")
        return 1

    per_data = data_time / n
    per_step = step_time / n
    steps_per_epoch = max(1, len(ds) // batch)
    est = steps_per_epoch * (per_data + per_step) / 60.0

    print(f"\n样本数 n={n}")
    print(f"取批   {per_data:.3f} s/batch")
    print(f"前反向 {per_step:.3f} s/batch")
    print(f"每 epoch 约 {steps_per_epoch} 步 -> 预计 {est:.1f} 分钟")
    if device.type == "cuda":
        print("峰值显存 %.2f GB" % (torch.cuda.max_memory_allocated(0) / 1e9))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
