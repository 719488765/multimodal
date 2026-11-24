"""生成 E3 配置：等优化步数、变窗口密度的受控实验。

设计依据（昨日诊断）：
- 时序臂在 MELD 上 train loss 降到 0.22，而片段臂 1.67，验证指标反而更低 -> 过拟合；
- 序列臂每 epoch 只有 1260 个非重叠窗口（stride=8），样本多样性远低于片段臂的 10085 话轮。

E3 保持"总优化步数"一致，只改变窗口密度：
- stride=8  -> 630 步/epoch，跑 8 epoch ≈ 5040 步
- stride=2  -> 2495 步/epoch，跑 2 epoch ≈ 4990 步
若多样性是瓶颈，stride=2 的验证 F1 应显著高于 stride=8。

另生成 F3（two_stage）修复后的重跑配置，用于补全融合对照表。

用法：python scripts/gen_e3_configs.py --out config/e3
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

BASE = Path("config/e2/f4_es_noseq_meld.yaml")
ROOT = Path("/mnt/sda1/lizhichun_24/code/multimodal/project")


def make(name: str, strategy: str, temporal: bool, window: int, stride: int,
         epochs: int, early_stop: bool) -> dict:
    cfg = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    cfg = copy.deepcopy(cfg)
    cfg["model"]["attention"]["fusion_strategy"] = strategy
    cfg.setdefault("data", {})["temporal"] = (
        {"enabled": True, "window": window, "stride": stride,
         "batch_size": 2, "use_previous_emotions": True}
        if temporal else {"enabled": False}
    )
    tr = cfg.setdefault("training", {})
    tr["seed"] = 42
    tr["num_epochs"] = epochs
    tr["early_stopping"] = {"enabled": early_stop, "monitor": "val_f1",
                            "patience": 3, "min_delta": 0.005}
    for mode in ("pretrain", "finetune"):
        block = tr.setdefault(mode, {})
        block["enabled"] = mode == "finetune"
        block["datasets"] = ["meld"]
        if mode == "finetune":
            block["epochs"] = epochs
    cfg["paths"] = {
        "checkpoint_dir": str(ROOT / "checkpoints_e3"),
        "log_dir": str(ROOT / "logs_e3"),
        "output_dir": str(ROOT / "outputs_e3"),
    }
    cfg["experiment"]["name"] = name
    cfg["experiment"]["use_tensorboard"] = True
    return cfg


SPECS = [
    # 名称, 策略, 时序, window, stride, epochs, early_stop
    ("E3_F3_two_stage_fixed_meld", "two_stage", False, 8, 8, 20, True),
    ("E3_G3_stride8_e8_meld", "emotion_shift", True, 8, 8, 8, False),
    ("E3_G3_stride2_e2_meld", "emotion_shift", True, 8, 2, 2, False),
]


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="config/e3")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, strategy, temporal, window, stride, epochs, early in SPECS:
        cfg = make(name, strategy, temporal, window, stride, epochs, early)
        p = out / f"{name}.yaml"
        p.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        steps = (10085 // 1) if not temporal else (10080 - window) // stride
        print(f"{p.name:34s} strategy={strategy:15s} temporal={temporal!s:5s} "
              f"stride={stride} epochs={epochs} early_stop={early}  约{steps * epochs}步")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
