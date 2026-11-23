"""判定标准融合路径能否学习：单 batch 过拟合测试。

若 30 步内 loss 明显下降 -> 模块可训练，F1 的差是别的原因；
若 loss 不动 -> 路径缺陷，需修 forward 契约。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.collate import multimodal_collate_fn  # noqa: E402
from data.dataset import MultimodalDataset  # noqa: E402
from models.multimodal_model import MultimodalEmotionModel  # noqa: E402
from scripts.train import _subset_by_datasets  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def run(cfg_path: str, steps: int = 30) -> None:
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    name = cfg["model"]["attention"]["fusion_strategy"]
    base = MultimodalDataset(cfg["data"]["root_dir"], split="val", config=cfg)
    base = _subset_by_datasets(base, cfg, "finetune")
    loader = DataLoader(base, batch_size=4, shuffle=True, num_workers=0,
                        collate_fn=multimodal_collate_fn)
    batch = next(iter(loader))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalEmotionModel(cfg).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    inputs = {}
    for k in ("text_input_ids", "text_attention_mask", "audio", "video"):
        if batch.get(k) is not None:
            inputs[k] = batch[k].to(device)
    labels = batch["emotion_label"].to(device).long().view(-1)

    print(f"[{name}] 单 batch 大小={labels.shape[0]} 类别={int(labels.max()) + 1}")
    for i in range(steps):
        out = model(**inputs)
        logits = out["emotion_logits"] if isinstance(out, dict) else out
        loss = torch.nn.functional.cross_entropy(logits.view(logits.size(0), -1), labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if i % 5 == 0 or i == steps - 1:
            print(f"  step {i:2d} loss={loss.item():.4f} logits.std={logits.std().item():.4f}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        run(p)
