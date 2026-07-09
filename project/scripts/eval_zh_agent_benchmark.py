#!/usr/bin/env python3
"""中文 Agent 情绪基准：校准 + 仲裁 + 语言路由（无需 GPU）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENT_ROOT = ROOT.parent / "emotion-agent" / "backend"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from utils.asr_emotion_calibration import apply_asr_emotion_calibration
from app.services.emotion_arbitration import arbitrate_emotion
from app.services.chinese_inference_router import build_inference_profile

EMOTION_NAMES = ["happy", "sad", "angry", "fear", "neutral", "anxious", "other"]
DEFAULT_CASES = ROOT / "data" / "agent_benchmark" / "zh_cases.json"


def _load_cases(path: Path) -> list:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"expected list in {path}")
    return data


def _model_emotion(case: dict) -> dict:
    probs = case.get("model_probs") or [1.0 / 7] * 7
    label = case.get("model_label") or EMOTION_NAMES[int(max(range(len(probs)), key=lambda i: probs[i]))]
    eid = EMOTION_NAMES.index(label) if label in EMOTION_NAMES else 4
    return {
        "emotion_id": eid,
        "emotion_label": label,
        "confidence": float(probs[eid]) if eid < len(probs) else 0.0,
        "valence": 0.0,
        "arousal": 0.0,
        "all_probs": list(probs),
    }


def run_benchmark(cases_path: Path, deploy_cfg: dict | None = None) -> int:
    deploy_cfg = deploy_cfg or {}
    cases = _load_cases(cases_path)
    failed = 0
    raw_ok = cal_ok = final_ok = 0

    for case in cases:
        text = case.get("text") or ""
        expect = case.get("expect_final") or case.get("expect_label")
        base = _model_emotion(case)

        profile = build_inference_profile(
            asr_text=text,
            user_text="",
            metadata={"text_source": "asr"},
            deploy_cfg=deploy_cfg,
        )

        raw_label = base["emotion_label"]
        if raw_label == expect:
            raw_ok += 1

        cal = apply_asr_emotion_calibration(dict(base), text, enabled=True)
        cal_label = cal.get("emotion_label")
        if cal_label == expect:
            cal_ok += 1

        cal["model_emotion_label"] = raw_label
        cal["model_emotion_id"] = base["emotion_id"]
        cal["model_confidence"] = base["confidence"]
        final = arbitrate_emotion(cal, text, asr_confidence=0.75)
        final_label = final.get("final_emotion_label") or final.get("emotion_label")

        if final_label == expect:
            final_ok += 1
            status = "OK"
        else:
            status = "FAIL"
            failed += 1

        print(
            f"{status} {case.get('name')}: raw={raw_label} cal={cal_label} "
            f"final={final_label} want={expect} lang={profile.get('language')} "
            f"skip_text={profile.get('skip_text_encoder')}"
        )

    n = len(cases)
    print(
        f"\nSummary n={n} raw={raw_ok}/{n} calibrated={cal_ok}/{n} "
        f"final={final_ok}/{n} failed={failed}"
    )
    return failed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument(
        "--deploy-config",
        type=Path,
        default=ROOT / "config" / "config_agent_deploy.yaml",
    )
    args = parser.parse_args()

    deploy = {}
    if args.deploy_config.is_file():
        import yaml

        deploy = yaml.safe_load(args.deploy_config.read_text(encoding="utf-8")) or {}

    return 1 if run_benchmark(args.cases, deploy) else 0


if __name__ == "__main__":
    raise SystemExit(main())
