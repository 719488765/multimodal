import base64
import io
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4] / "project"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.emotion_inference_service import (  # noqa: E402
    _frames_list_to_tensor,
    decode_base64_bytes,
    preprocess_video_from_bytes,
    strip_base64_payload,
)


def test_strip_base64_data_url() -> None:
    raw = base64.b64encode(b"hello").decode("ascii")
    wrapped = f"data:image/jpeg;base64,{raw}"
    assert strip_base64_payload(wrapped) == raw
    assert decode_base64_bytes(wrapped) == b"hello"


def test_frames_list_replicate_single_frame() -> None:
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    tensor = _frames_list_to_tensor([frame], num_frames=4)
    assert tensor.shape == (1, 4, 3, 8, 8)


def test_preprocess_video_from_bytes() -> None:
    from PIL import Image

    config = {"data": {"video": {"frame_size": 112, "num_frames": 4}}}
    img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    tensor = preprocess_video_from_bytes(buf.getvalue(), config)
    assert tensor is not None
    assert tensor.shape[0] == 1
    assert tensor.shape[1] == 4


@pytest.mark.gpu
def test_load_ap2_m1_checkpoint() -> None:
    """Optional: RUN_GPU_TESTS=1 pytest -m gpu"""
    import os

    if os.environ.get("RUN_GPU_TESTS") != "1":
        pytest.skip("Set RUN_GPU_TESTS=1 to run GPU checkpoint load test")

    from utils.emotion_inference_service import EmotionInferenceService

    ckpt = (
        PROJECT_ROOT
        / "checkpoints_accuracy_seq/AP2_M1_ES_3ds_effbatch8_s3407_20260422_210615"
        / "checkpoint_pretrain_best_f1.pth"
    )
    cfg = PROJECT_ROOT / "config/rerun/accuracy_plan/ap2_M1_effbatch8_ES_3ds_s3407.yaml"
    if not ckpt.is_file():
        pytest.skip(f"checkpoint missing: {ckpt}")

    service = EmotionInferenceService(str(cfg), str(ckpt), device="cuda")
    service.load()
    health = service.health()
    assert health["loaded"] is True
