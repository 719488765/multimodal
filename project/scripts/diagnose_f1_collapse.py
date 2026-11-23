"""诊断标准融合臂的预测冻结：加载已训练 checkpoint，统计预测类别分布。

判据：若某单一类别占全部预测的绝大多数，且与真实分布不符，即为塌缩。
"""
from __future__ import annotations

import sys
from collections import Counter
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
    cfg_path = sys.argv[1]
    ckpt_path = sys.argv[2]
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    cfg.setdefault("training", {}).setdefault("finetune", {})["datasets"] = ["meld"]
    _apply_single_domain_emotion_classes(cfg, "finetune")
    print("emotion_classes =", cfg["model"]["output"]["emotion_classes"])

    base = MultimodalDataset(cfg["data"]["root_dir"], split="val", config=cfg)
    base = _subset_by_datasets(base, cfg, "finetune")
    loader = DataLoader(base, batch_size=16, shuffle=False, num_workers=0,
                        collate_fn=multimodal_collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalEmotionModel(cfg).to(device)
    sd = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(sd.get("model_state_dict", sd), strict=False)
    model.eval()

    preds, tgts = [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            inputs = _batch_to_model_inputs(batch, device, cfg, use_prev_labels=False)
            out = model(**inputs)
            logits = out["emotion_logits"]
            preds.extend(logits.argmax(-1).cpu().tolist())
            tgts.extend(batch["emotion_label"].view(-1).cpu().tolist())
            if i >= 20:
                break

    pc, tc = Counter(preds), Counter(tgts)
    print("样本数", len(preds))
    print("预测分布", dict(sorted(pc.items())))
    print("真实分布", dict(sorted(tc.items())))
    top = pc.most_common(1)[0]
    print("最大单一预测占比 %.3f" % (top[1] / len(preds)))
    print("不同预测类别数", len(pc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
