from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class IngestChunkRequest(BaseModel):
    session_id: str
    client_ts: float = Field(description="Unix timestamp from client")
    video_chunk_b64: str = ""
    audio_chunk_b64: str = ""


class IngestChunkResponse(BaseModel):
    session_id: str
    accepted: bool
    buffered_windows: int


class EmotionInferRequest(BaseModel):
    session_id: str
    text: str = ""
    video_chunk_b64: str = ""
    audio_chunk_b64: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EmotionInferResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    emotion_label: str
    confidence: float
    valence: float
    arousal: float
    all_probs: List[float]
    degraded_mode: bool
    model_provider: str
    # 用于区分真实 checkpoint 与 mock（见 README_DEPLOY §如何确认使用训练模型）
    inference_source: str = Field(
        default="",
        description="checkpoint=训练权重推理; mock_heuristic=关键词随机; mock_fallback=真实推理失败后降级",
    )
    checkpoint_preset: str = ""
    fusion_strategy: str = ""
    inference_ms: float = 0.0
    emotion_id: int = -1
    checkpoint_file: str = ""
    asr_text: str = ""
    asr_confidence: float = 0.0
    asr_error: str = ""
    asr_provider: str = ""
    top_emotions: List[Dict[str, Any]] = Field(default_factory=list)
    all_probs_labeled: List[Dict[str, Any]] = Field(default_factory=list)
    pipeline_trace: Dict[str, Any] = Field(default_factory=dict)
    temporal_windows: List[Dict[str, Any]] = Field(default_factory=list)
    temporal_summary: Dict[str, Any] = Field(default_factory=dict)
    per_window_probs: List[Dict[str, Any]] = Field(default_factory=list)
    video_decode_mode: str = ""
    asr_calibration_applied: bool = False
    asr_calibration_reason: str = ""
    calibration_profile: str = ""
    model_probs_before_calibration: List[float] = Field(default_factory=list)
    final_emotion_label: str = ""
    final_emotion_id: int = -1
    final_confidence: float = 0.0
    arbitration_source: str = ""
    arbitration_reason: str = ""
    model_emotion_label: str = ""
    model_confidence: float = 0.0
    is_flat_distribution: bool = False


class AgentRespondRequest(BaseModel):
    session_id: str
    emotion_label: str
    confidence: float
    context_text: str = ""
    valence: Optional[float] = None
    arousal: Optional[float] = None
    all_probs: List[float] = Field(default_factory=list)
    all_probs_labeled: List[Dict[str, Any]] = Field(default_factory=list)
    top_emotions: List[Dict[str, Any]] = Field(default_factory=list)


class AgentRespondResponse(BaseModel):
    reply_text: str
    tone: str
    safe_mode: bool
    llm_provider: str
    llm_model: str = ""
    llm_error: str = ""


class SessionEvent(BaseModel):
    session_id: str
    ts: float
    event_type: str
    payload: Dict[str, Any]
