"""
推理脚本（CLI 薄封装，与 agent 共用 EmotionInferenceService）
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.emotion_inference_service import EMOTION_NAMES, EmotionInferenceService


def main():
    parser = argparse.ArgumentParser(description="Inference with Multimodal Emotion Recognition Model")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--video", type=str, default=None, help="Path to video file")
    parser.add_argument("--audio", type=str, default=None, help="Path to audio file")
    parser.add_argument("--physiological", type=str, default=None, help="Path to physiological signal file")
    parser.add_argument("--text", type=str, default=None, help="Text input")
    parser.add_argument("--input_dir", type=str, default=None, help="Directory containing input files")
    parser.add_argument("--device", type=str, default=None, help="Override device (cuda/cpu)")
    args = parser.parse_args()

    service = EmotionInferenceService(
        config_path=args.config,
        checkpoint_path=args.model_path,
        device=args.device,
    )
    service.load()
    print(f"Model loaded from {args.model_path}")

    if args.input_dir:
        print(f"Processing directory: {args.input_dir} (not implemented)")
        return

    results = service.predict_from_paths(
        video_path=args.video,
        audio_path=args.audio,
        physiological_path=args.physiological,
        text=args.text,
    )

    print("\n=== Inference Results ===")
    print(f"Emotion: {results['emotion']} (ID: {results['emotion_id']})")
    print(f"Confidence: {results['confidence']:.4f}")
    print(f"Valence: {results['valence']:.4f}")
    print(f"Arousal: {results['arousal']:.4f}")
    print("\nAll Emotion Probabilities:")
    for name, prob in zip(EMOTION_NAMES, results["all_probs"]):
        print(f"  {name}: {prob:.4f}")


if __name__ == "__main__":
    main()
