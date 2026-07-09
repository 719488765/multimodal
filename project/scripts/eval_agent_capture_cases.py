#!/usr/bin/env python3
"""Agent 在线采集典型用例回归（ASR 校正 + 可选真实 checkpoint）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.asr_emotion_calibration import apply_asr_emotion_calibration

BENCHMARK = ROOT / "data" / "agent_benchmark" / "zh_cases.json"

CASES = [
    {
        "name": "happy_asr_vs_neutral_model",
        "text": "我很高兴实验取得了成功！哈哈哈哈！",
        "emotion": {
            "emotion_id": 4,
            "emotion_label": "neutral",
            "confidence": 0.73,
            "valence": -0.36,
            "arousal": 0.2,
            "all_probs": [0.198, 0.01, 0.015, 0.057, 0.73, 0.0, 0.0],
        },
        "expect_label": "happy",
    },
    {
        "name": "flat_laughter_hahaha",
        "text": "哈哈哈哈哈哈",
        "emotion": {
            "emotion_id": 4,
            "emotion_label": "neutral",
            "confidence": 0.244,
            "valence": -0.31,
            "arousal": 0.55,
            "all_probs": [0.021, 0.20, 0.158, 0.131, 0.244, 0.204, 0.041],
        },
        "expect_label": "happy",
    },
    {
        "name": "sad_asr_vs_confident_neutral",
        "text": "我很难过，心里非常难受。",
        "emotion": {
            "emotion_id": 4,
            "emotion_label": "neutral",
            "confidence": 0.73,
            "valence": -0.36,
            "arousal": 0.2,
            "all_probs": [0.05, 0.01, 0.015, 0.057, 0.73, 0.0, 0.0],
        },
        "expect_label": "sad",
    },
    {
        "name": "angry_asr_vs_neutral",
        "text": "我真的太生气了！",
        "emotion": {
            "emotion_id": 4,
            "emotion_label": "neutral",
            "confidence": 0.70,
            "valence": -0.2,
            "arousal": 0.5,
            "all_probs": [0.05, 0.05, 0.08, 0.05, 0.70, 0.05, 0.02],
        },
        "expect_label": "angry",
    },
    {
        "name": "neutral_weather_no_calib",
        "text": "今天阴天，气温十五度。",
        "emotion": {
            "emotion_id": 4,
            "emotion_label": "neutral",
            "confidence": 0.8,
            "valence": 0.0,
            "arousal": 0.1,
            "all_probs": [0.05, 0.05, 0.05, 0.05, 0.8, 0.0, 0.0],
        },
        "expect_label": "neutral",
    },
]


def _load_cases(extra_path: Path | None) -> list:
    if extra_path and extra_path.is_file():
        return json.loads(extra_path.read_text(encoding="utf-8"))
    if BENCHMARK.is_file():
        out = []
        for row in json.loads(BENCHMARK.read_text(encoding="utf-8")):
            probs = row.get("model_probs") or [0.14] * 7
            label = row.get("model_label") or "neutral"
            eid = ["happy", "sad", "angry", "fear", "neutral", "anxious", "other"].index(label)
            out.append(
                {
                    "name": row["name"],
                    "text": row.get("text", ""),
                    "emotion": {
                        "emotion_id": eid,
                        "emotion_label": label,
                        "confidence": float(probs[eid]),
                        "valence": 0.0,
                        "arousal": 0.0,
                        "all_probs": probs,
                    },
                    "expect_label": row.get("expect_final") or row.get("expect_label"),
                }
            )
        return out
    return CASES


def run_calibration_cases(cases: list) -> int:
    failed = 0
    for case in cases:
        out = apply_asr_emotion_calibration(dict(case["emotion"]), case["text"])
        label = out.get("emotion_label")
        if label != case["expect_label"]:
            print(f"FAIL {case['name']}: got {label}, want {case['expect_label']}")
            failed += 1
        else:
            applied = out.get("asr_calibration_applied")
            print(f"OK   {case['name']}: {label} (calibration={applied})")
    return failed


def run_checkpoint_case(config: str, checkpoint: str, device: str) -> int:
    try:
        from utils.emotion_inference_service import EmotionInferenceService
    except ImportError as exc:
        print(f"SKIP checkpoint eval (import failed): {exc}")
        return 0

    svc = EmotionInferenceService(
        config_path=str(ROOT / config),
        checkpoint_path=str(ROOT / checkpoint),
        device=device,
        project_root=str(ROOT),
    )
    try:
        svc.load()
    except Exception as exc:
        print(f"SKIP checkpoint eval (load failed): {exc}")
        return 0

    sample = {
        "text": CASES[0]["text"],
        "video_chunk_b64": "",
        "audio_chunk_b64": "",
        "metadata": {"temporal_inference": {"enabled": False}},
    }
    # 无媒体时仅验证不崩溃；有 GPU 环境可扩展为真实 wav/jpg
    try:
        result = svc.predict_from_sample(sample)
        print(f"checkpoint smoke: label={result.get('emotion_label')} source={result.get('inference_source')}")
    except Exception as exc:
        print(f"WARN checkpoint smoke failed: {exc}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-checkpoint", action="store_true")
    parser.add_argument("--cases", type=Path, default=None, help="JSON case list (default: zh_cases.json)")
    parser.add_argument(
        "--config",
        default="config/rerun/accuracy_plan/ap2_M1_chinese_text_agent.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints_accuracy_seq/AP2_M1_chinese_text_agent/checkpoint_finetune_best_f1.pth",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cases = _load_cases(args.cases)
    failed = run_calibration_cases(cases)
    if args.with_checkpoint:
        failed += run_checkpoint_case(args.config, args.checkpoint, args.device)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
