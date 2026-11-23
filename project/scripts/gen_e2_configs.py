"""生成 E2 受控实验配置。

设计原则：所有臂从同一模板深拷贝，只改被测变量（fusion_strategy / temporal），
其余（seed、epochs、batch、数据过滤、路径）完全一致，避免 R4 中"ES 组额外重训、
基线没重训"的预算不对等重演。

用法：
    python scripts/gen_e2_configs.py --out config/e2 --epochs 8
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

BASE = Path("config/config_AVT_noDA_emotion_shift.yaml")
ROOT = Path("/mnt/sda1/lizhichun_24/code/multimodal/project")

# (文件名, 臂标签, fusion_strategy, temporal_enabled, use_prev, leader)
ARMS = [
    ("f1_standard_meld",      "F1_standard",     "standard",        False, None,  None),
    ("f2_lf_text_meld",       "F2_lf_text",      "leader_follower", False, None,  "text"),
    ("f3_two_stage_meld",     "F3_two_stage",    "two_stage",       False, None,  None),
    ("f4_es_noseq_meld",      "F4_es_noseq",     "emotion_shift",   False, None,  None),
    ("g2_es_seq_noprev_meld", "G2_es_seq_noprev","emotion_shift",   True,  False, None),
    ("g3_es_seq_prev_meld",   "G3_es_seq_prev",  "emotion_shift",   True,  True,  None),
    ("c_f1_standard_crema",   "F1_standard",     "standard",        False, None,  None),
    ("c_f4_es_noseq_crema",   "F4_es_noseq",     "emotion_shift",   False, None,  None),
    ("c_g3_es_seq_prev_crema","G3_es_seq_prev",  "emotion_shift",   True,  True,  None),
]


def build(base: dict, arm: str, strategy: str, temporal: bool, use_prev, leader,
          epochs: int, dataset: str, window: int, stride: int, seq_batch: int) -> dict:
    cfg = copy.deepcopy(base)

    cfg["model"]["attention"]["fusion_strategy"] = strategy
    if leader:
        cfg["model"]["attention"]["leader_modal"] = leader

    cfg.setdefault("data", {})["temporal"] = (
        {
            "enabled": True,
            "window": window,
            "stride": stride,
            "batch_size": seq_batch,
            "use_previous_emotions": bool(use_prev),
        }
        if temporal
        else {"enabled": False}
    )

    tr = cfg.setdefault("training", {})
    tr["seed"] = 42
    tr["num_epochs"] = epochs
    tr["batch_size"] = 1
    # 收敛判据：不同融合路径的收敛速度差异很大，按“相同轮数”比较会误判。
    # 统一改为按验证集 Macro-F1 是否仍在提升来停止（上限 epochs 轮）。
    tr["early_stopping"] = {
        "enabled": True,
        "monitor": "val_f1",
        "patience": 3,
        "min_delta": 0.005,
    }
    for mode in ("pretrain", "finetune"):
        block = tr.setdefault(mode, {})
        block["enabled"] = mode == "finetune"
        block["datasets"] = [dataset]
        if mode == "finetune":
            block["epochs"] = epochs

    cfg["paths"] = {
        "checkpoint_dir": str(ROOT / "checkpoints_e2"),
        "log_dir": str(ROOT / "logs_e2"),
        "output_dir": str(ROOT / "outputs_e2"),
    }
    cfg["experiment"]["name"] = f"E2_{arm}_{dataset}"
    cfg["experiment"]["use_tensorboard"] = True
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="config/e2")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--seq-batch", type=int, default=2)
    ap.add_argument("--only", default="", help="逗号分隔的臂标签过滤，如 F1_standard,F4_es_noseq")
    args = ap.parse_args()

    base = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    keep = {s.strip() for s in args.only.split(",") if s.strip()}
    made = []
    for fname, arm, strategy, temporal, use_prev, leader in ARMS:
        if keep and arm not in keep:
            continue
        dataset = "crema" if fname.startswith("c_") else "meld"
        cfg = build(base, arm, strategy, temporal, use_prev, leader,
                    args.epochs, dataset, args.window, args.stride, args.seq_batch)
        path = out / f"{fname}.yaml"
        path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        made.append((path.name, cfg["experiment"]["name"], strategy, temporal, use_prev, dataset))

    print(f"generated {len(made)} configs -> {out}")
    for row in made:
        print("  %-28s name=%-28s strategy=%-16s temporal=%-5s prev=%-5s ds=%s" % row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
