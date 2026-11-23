"""判定塌缩位置：比较 fused_features 与 emotion_logits 的样本间方差。

若 fused_features 在样本维度上几乎无差异 -> 塌缩发生在融合层；
若 fused_features 有差异而 logits 无差异 -> 塌缩发生在分类头。
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
from scripts.train import _apply_single_domain_emotion_classes, _batch_to_model_inputs, _subset_by_datasets  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def main() -> int:
    cfg_path, ckpt_path = sys.argv[1], sys.argv[2]
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    cfg.setdefault("training", {}).setdefault("finetune", {})["datasets"] = ["meld"]
    _apply_single_domain_emotion_classes(cfg, "finetune")

    base = MultimodalDataset(cfg["data"]["root_dir"], split="val", config=cfg)
    base = _subset_by_datasets(base, cfg, "finetune")
    loader = DataLoader(base, batch_size=8, shuffle=False, num_workers=0,
                        collate_fn=multimodal_collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalEmotionModel(cfg).to(device)
    sd = {} if ckpt_path == "fresh" else torch.load(ckpt_path, map_location=device)
    model.load_state_dict(sd.get("model_state_dict", sd), strict=False)
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(loader):
            inputs = _batch_to_model_inputs(batch, device, cfg, use_prev_labels=False)
            out = model(**inputs)
            ff = out["fused_features"]
            lg = out["emotion_logits"]
            # 样本间方差（沿 batch 维求 std，再对特征维取均值）
            ff_std = ff.std(dim=0).mean().item()
            ff_absmean = ff.abs().mean().item()
            # 每个特征维在样本间是否几乎恒定
            ff_dim_std = ff.std(dim=0)
            dead = int((ff_dim_std < 1e-4).sum().item())
            lg_std = lg.std(dim=0).mean().item()
            # 逐样本之间的余弦相似度
            fn = torch.nn.functional.normalize(ff, dim=-1)
            sim = (fn @ fn.t())
            off = sim[~torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)]
            print(f"batch {i}: |fused|={ff_absmean:.4f}  fused样本间std={ff_std:.6f}  "
                  f"恒定维度={dead}/{ff.size(-1)}  logits样本间std={lg_std:.6f}  "
                  f"样本两两余弦均值={off.mean().item():.4f}")
            if i >= 2:
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
