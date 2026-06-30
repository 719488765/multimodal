# Author: AI
# Date: 2026-04-11
# Description: 打印准确率优化计划各阶段建议训练命令（工作目录应为 project/）

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAN_DIR = os.path.join(ROOT, "config", "rerun", "accuracy_plan")


def main():
    yamls = sorted(
        f
        for f in os.listdir(PLAN_DIR)
        if f.endswith(".yaml") and f != "ap4_da_sweep_manifest.yaml"
    )
    print("# cd project")
    print("# 日志与 checkpoint 目录：logs_accuracy_seq/、checkpoints_accuracy_seq/（与 logs_rerun 隔离）")
    print("# conda activate <your_env>   # 确保 torch / cv2 / 数据就绪")
    print()
    for name in yamls:
        rel = f"config/rerun/accuracy_plan/{name}"
        print(f"python3 scripts/train.py --config {rel} --mode pretrain")
    print()
    print("# 阶段4 DA：使用已改路径的副本，例如：")
    print("python3 scripts/train.py --config config/rerun/accuracy_plan/ap4_config_AVT_DA_w005_accuracy_seq.yaml --mode pretrain")
    print()
    print("# 指标：优先看 metrics.csv 中 val 的 f1 峰值与 checkpoint_pretrain_best_f1.pth；")
    print("# 若已启用 cls_ce_unweighted 列，可对照 train/val 不加权 CE _gap。")


if __name__ == "__main__":
    main()
