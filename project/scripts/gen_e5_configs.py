"""生成 E5 配置：为 F1/F2/F4 各补两个随机种子（1337、2024）。

除 seed 与输出目录外，其余超参、数据过滤、收敛判据与 E2 完全一致，
使"三 seed 均值±标准差"可与单 seed 结果直接对照。
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path("/mnt/sda1/lizhichun_24/code/multimodal/project")
BASES = {
    "F1_standard": ROOT / "config/e2/f1_standard_meld.yaml",
    "F2_lf_text": ROOT / "config/e2/f2_lf_text_meld.yaml",
    "F4_es_noseq": ROOT / "config/e2/f4_es_noseq_meld.yaml",
}
SEEDS = [1337, 2024]


def main() -> int:
    out = ROOT / "config/e5"
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for arm, path in BASES.items():
        base = yaml.safe_load(path.read_text(encoding="utf-8"))
        for seed in SEEDS:
            cfg = copy.deepcopy(base)
            cfg["training"]["seed"] = seed
            cfg["training"]["num_epochs"] = 20
            cfg["training"]["early_stopping"] = {
                "enabled": True, "monitor": "val_f1", "patience": 3, "min_delta": 0.005,
            }
            for mode in ("pretrain", "finetune"):
                block = cfg["training"].setdefault(mode, {})
                block["enabled"] = mode == "finetune"
                block["datasets"] = ["meld"]
                if mode == "finetune":
                    block["epochs"] = 20
            cfg["paths"] = {
                "checkpoint_dir": str(ROOT / "checkpoints_e5"),
                "log_dir": str(ROOT / "logs_e5"),
                "output_dir": str(ROOT / "outputs_e5"),
            }
            cfg["experiment"]["name"] = f"E5_{arm}_seed{seed}_meld"
            (out / f"{arm}_seed{seed}.yaml").write_text(
                yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
            print(f"wrote {arm}_seed{seed}.yaml")
            n += 1
    print(f"total {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
