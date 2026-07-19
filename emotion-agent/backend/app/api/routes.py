from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect

from app.models.schemas import (
    AgentRespondRequest,
    AgentRespondResponse,
    EmotionInferRequest,
    EmotionInferResponse,
    IngestChunkRequest,
    IngestChunkResponse,
)
from app.services.upload_buffer import assemble as assemble_upload, put_chunk
from app.services.asr_service import ASRService
from app.services.ingest_buffer import IngestBuffer
from app.services.llm_service import LLMService
from app.core.config import CHINESE_BERT_PRESETS, list_available_presets, settings
from app.services.chinese_inference_router import build_inference_profile
from app.services.emotion_arbitration import arbitrate_emotion
from app.services.model_router import ModelRouter
from app.services.session_store import SessionStore
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v1")
ingest_buffer = IngestBuffer(window_size_sec=3.0, step_sec=1.0)
_model_router: Optional[ModelRouter] = None
asr_service = ASRService()
llm_service = LLMService()
session_store = SessionStore()
ws_clients: Dict[str, List[WebSocket]] = defaultdict(list)
logger = logging.getLogger(__name__)


class ModelPreloadRequest(BaseModel):
    preset: str = Field(..., description="CHECKPOINT_PRESETS id to warm-load")


def get_model_router() -> ModelRouter:
    global _model_router
    if _model_router is None:
        _model_router = ModelRouter()
    return _model_router


def _load_deploy_postprocess_config() -> Dict[str, Any]:
    """读取 config_agent_deploy.yaml 中的后处理配置。"""
    root = Path(settings.project_root).resolve()
    deploy_cfg = root / "config" / "config_agent_deploy.yaml"
    if not deploy_cfg.is_file():
        return {}
    try:
        import yaml

        with open(deploy_cfg, "r", encoding="utf-8") as f:
            deploy = yaml.safe_load(f) or {}
        return deploy
    except Exception:
        return {}


def _apply_asr_calibration(
    emotion: Dict[str, Any],
    merged_text: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """调用 project 内 ASR–情绪一致性校正（可配置关闭）。"""
    import sys
    from pathlib import Path

    meta = metadata or {}
    if meta.get("asr_calibration") is False:
        emotion["asr_calibration_applied"] = False
        emotion["asr_calibration_reason"] = "disabled_by_metadata"
        return emotion

    root = Path(settings.project_root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    enabled = True
    deploy = _load_deploy_postprocess_config()
    cal = deploy.get("asr_emotion_calibration") or {}
    enabled = bool(cal.get("enabled", True))

    from utils.asr_emotion_calibration import apply_asr_emotion_calibration

    label_before = emotion.get("emotion_label")
    apply_asr_emotion_calibration(emotion, merged_text, enabled=enabled)
    if emotion.get("asr_calibration_applied"):
        logger.info(
            "[asr_calibration] %s -> %s reason=%s",
            label_before,
            emotion.get("emotion_label"),
            emotion.get("asr_calibration_reason"),
        )
    return emotion


def startup_router() -> None:
    """Load emotion model at application startup (GPU checkpoint)."""
    get_model_router()
    asr_status = asr_service.probe()
    if asr_status.get("ok"):
        logger.info("[ASR] startup ok: %s", asr_status.get("message"))
    else:
        logger.warning("[ASR] startup NOT ready: %s | %s", asr_status.get("message"), asr_status.get("hint", ""))
    llm_status = llm_service.probe()
    if llm_status.get("ok"):
        logger.info("[LLM] startup ok: %s", llm_status.get("message"))
    else:
        logger.warning("[LLM] startup NOT ready: %s | %s", llm_status.get("message"), llm_status.get("hint", ""))


def shutdown_router() -> None:
    global _model_router
    _model_router = None


async def _broadcast(session_id: str, event: dict) -> None:
    data = json.dumps(event, ensure_ascii=False)
    active = []
    for ws in ws_clients[session_id]:
        try:
            await ws.send_text(data)
            active.append(ws)
        except Exception:
            continue
    ws_clients[session_id] = active


@router.get("/health")
def health() -> dict:
    asr_status = asr_service.probe()
    llm_status = llm_service.probe()
    model_health = get_model_router().health()
    using_trained = (
        model_health.get("provider") == "current" and model_health.get("loaded") is True
    )
    return {
        "ok": True,
        "model": model_health,
        "using_trained_checkpoint": using_trained,
        "asr": asr_status,
        "asr_ok": bool(asr_status.get("ok")),
        "llm": llm_status,
        "llm_ok": bool(llm_status.get("ok")),
    }


@router.get("/model/status")
def model_status(all: bool = False) -> dict:
    """详细模型绑定状态，用于确认是否使用训练 checkpoint（非 mock）。"""
    router_ = get_model_router()
    model_health = router_.health()
    using_trained = (
        settings.model_provider.lower() == "current"
        and model_health.get("loaded") is True
        and model_health.get("provider") == "current"
    )
    available_presets = list_available_presets(include_hidden=bool(all))
    return {
        "ok": True,
        "using_trained_checkpoint": using_trained,
        "model_provider_env": settings.model_provider,
        "checkpoint_preset_env": settings.model_checkpoint_preset,
        "identification": {
            "trained": "inference_source=checkpoint 且 model_provider=current 且 loaded=true",
            "mock": "inference_source=mock_heuristic 或 model_provider=mock",
            "fallback": "inference_source=mock_fallback（真实推理失败后降级）",
        },
        "available_presets": available_presets,
        "model": model_health,
    }


@router.post("/model/preload")
def model_preload(payload: ModelPreloadRequest) -> dict:
    """预热加载指定 checkpoint preset，减少切换后首次推理延迟。"""
    return get_model_router().preload(payload.preset)


@router.post("/ingest/chunk", response_model=IngestChunkResponse)
async def ingest_chunk(payload: IngestChunkRequest) -> IngestChunkResponse:
    ingest_buffer.push_chunk(
        session_id=payload.session_id,
        client_ts=payload.client_ts,
        video_chunk_b64=payload.video_chunk_b64,
        audio_chunk_b64=payload.audio_chunk_b64,
    )
    windows = ingest_buffer.build_windows(payload.session_id)
    session_store.add(payload.session_id, "ingest", {"windows": len(windows)})
    await _broadcast(payload.session_id, {"event": "ingest", "windows": len(windows)})
    return IngestChunkResponse(session_id=payload.session_id, accepted=True, buffered_windows=len(windows))


async def _emotion_infer_core(
    session_id: str,
    text: str,
    video_chunk_b64: str,
    audio_chunk_b64: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> EmotionInferResponse:
    print(
        "[emotion_infer]",
        {
            "session_id": session_id,
            "text_len": len(text or ""),
            "has_video": bool(video_chunk_b64),
            "has_audio": bool(audio_chunk_b64),
            "audio_len": len(audio_chunk_b64 or ""),
            "upload": (metadata or {}).get("upload", "json"),
        },
    )
    logger.info(
        "emotion_infer session=%s text_len=%d has_video=%s has_audio=%s audio_len=%d upload=%s",
        session_id,
        len(text or ""),
        bool(video_chunk_b64),
        bool(audio_chunk_b64),
        len(audio_chunk_b64 or ""),
        (metadata or {}).get("upload", "json"),
    )
    audio_raw = (metadata or {}).get("audio_raw")
    if isinstance(audio_raw, (bytes, bytearray)) and len(audio_raw) > 0:
        asr_result = await asyncio.to_thread(asr_service.transcribe_bytes, bytes(audio_raw), "wav")
    else:
        asr_result = await asyncio.to_thread(asr_service.transcribe, audio_chunk_b64)
    print(
        "[asr_result]",
        {
            "session_id": session_id,
            "provider": asr_result.get("provider", ""),
            "confidence": asr_result.get("confidence", 0.0),
            "text_len": len(asr_result.get("text", "") or ""),
            "error": asr_result.get("error", ""),
        },
    )
    logger.info(
        "asr_result session=%s provider=%s confidence=%s text_len=%d error=%s",
        session_id,
        asr_result.get("provider", ""),
        asr_result.get("confidence", 0.0),
        len(asr_result.get("text", "") or ""),
        asr_result.get("error", ""),
    )
    merged_text = text or asr_result.get("text", "")
    text_source = "user_input" if (text or "").strip() else ("asr" if (asr_result.get("text") or "").strip() else "empty")
    deploy = _load_deploy_postprocess_config()
    meta = dict(metadata or {})
    meta["text_source"] = text_source
    inference_profile = build_inference_profile(
        asr_text=asr_result.get("text", "") or "",
        user_text=text or "",
        metadata=meta,
        deploy_cfg=deploy,
    )
    meta["inference_profile"] = inference_profile
    sample: Dict[str, Any] = {
        "session_id": session_id,
        "text": merged_text,
        "video_chunk_b64": video_chunk_b64,
        "audio_chunk_b64": audio_chunk_b64,
        "metadata": meta,
    }

    if not sample.get("video_chunk_b64") or not sample.get("audio_chunk_b64"):
        windows = ingest_buffer.build_windows(session_id)
        if windows:
            win = windows[-1]
            sample["video_chunk_b64"] = sample.get("video_chunk_b64") or win.get("video_chunk_b64", "")
            sample["audio_chunk_b64"] = sample.get("audio_chunk_b64") or win.get("audio_chunk_b64", "")

    emotion = get_model_router().infer(sample)
    model_label = emotion.get("emotion_label")
    model_confidence = float(emotion.get("confidence") or 0.0)
    model_emotion_id = int(emotion.get("emotion_id", 4))
    emotion["model_emotion_label"] = model_label
    emotion["model_emotion_id"] = model_emotion_id
    emotion["model_confidence"] = model_confidence

    # 中文 BERT 微调轨：文本已进模型，禁止词典校准/强仲裁把结果改成「像纯文本分类」
    used_preset = str(emotion.get("checkpoint_preset") or meta.get("checkpoint_preset") or "").lower()
    trust_multimodal = used_preset in CHINESE_BERT_PRESETS

    label_before_calibration = model_label
    if trust_multimodal:
        emotion["asr_calibration_applied"] = False
        emotion["asr_calibration_reason"] = "disabled_for_chinese_bert_avt"
        label_after_calibration = model_label
    else:
        _apply_asr_calibration(emotion, merged_text, metadata)
        label_after_calibration = emotion.get("emotion_label")

    deploy = _load_deploy_postprocess_config()
    arb_cfg = deploy.get("emotion_arbitration") or {}
    flat_thr = float(
        inference_profile.get("flat_threshold")
        or arb_cfg.get("flat_threshold")
        or 0.38
    )
    if trust_multimodal:
        # 仅在模型分布极平时才允许轻量仲裁；默认完全信任 AVT checkpoint
        flat_thr = float(arb_cfg.get("chinese_bert_flat_threshold", 0.32))
    arbitrate_emotion(
        emotion,
        merged_text,
        float(asr_result.get("confidence", 0.0) or 0.0),
        flat_threshold=flat_thr,
        low_conf_threshold=float(arb_cfg.get("low_conf_threshold", 0.42)),
        neutral_override_threshold=float(arb_cfg.get("neutral_override_threshold", 0.55)),
        trust_model=trust_multimodal,
    )
    trace = emotion.get("pipeline_trace") or {}
    trace_steps: List[Dict[str, Any]] = [
        {
            "stage": "1_ingest",
            "status": "ok",
            "detail": f"session={session_id} upload={(metadata or {}).get('upload', 'json')}",
            "video_bytes": (metadata or {}).get("video_byte_count") or len(video_chunk_b64 or ""),
            "audio_bytes": (metadata or {}).get("audio_byte_count") or len(audio_chunk_b64 or ""),
        },
        {
            "stage": "2_asr",
            "status": (
                "error"
                if asr_result.get("error")
                else ("mock" if asr_result.get("provider") == "mock" else "ok")
            ),
            "provider": asr_result.get("provider", ""),
            "text_len": len(asr_result.get("text", "") or ""),
            "confidence": asr_result.get("confidence", 0.0),
            "text_preview": (asr_result.get("text", "") or "")[:60],
            "error": asr_result.get("error", ""),
        },
        {
            "stage": "2_language_detect",
            "status": "ok",
            "language": inference_profile.get("language", "unknown"),
            "skip_text": inference_profile.get("skip_text_encoder", False),
            "leader_override": inference_profile.get("leader_override"),
        },
        {
            "stage": "3_text_merge",
            "status": "ok" if merged_text.strip() else "empty",
            "source": text_source,
            "merged_len": len(merged_text),
            "merged_preview": merged_text[:60],
        },
        {
            "stage": "4_emotion_model",
            "status": (
                "ok"
                if str(emotion.get("inference_source", "")).startswith("checkpoint")
                else emotion.get("inference_source", "unknown")
            ),
            "provider": emotion.get("model_provider", ""),
            "inference_source": emotion.get("inference_source", ""),
            "label": model_label,
            "confidence": model_confidence,
            "inference_ms": emotion.get("inference_ms", 0.0),
            "is_mock": emotion.get("inference_source", "").startswith("mock"),
            "num_windows": (emotion.get("temporal_summary") or {}).get("num_windows"),
            "mode": "temporal" if emotion.get("temporal_windows") else "single",
            "video_decode_mode": emotion.get("video_decode_mode"),
            "preset": emotion.get("checkpoint_preset"),
        },
        {
            "stage": "5_asr_calibration",
            "status": "applied" if emotion.get("asr_calibration_applied") else "skipped",
            "label_before": label_before_calibration,
            "label_after": label_after_calibration,
            "reason": emotion.get("asr_calibration_reason", ""),
            "profile": emotion.get("calibration_profile", ""),
        },
        {
            "stage": "6_arbitration",
            "status": "applied" if emotion.get("arbitration_source") != "model" else "passthrough",
            "model_label": emotion.get("model_emotion_label"),
            "final_label": emotion.get("final_emotion_label"),
            "source": emotion.get("arbitration_source", ""),
            "reason": emotion.get("arbitration_reason", ""),
            "flat": emotion.get("is_flat_distribution"),
        },
    ]
    trace["steps"] = trace_steps
    trace["asr_calibration"] = {
        "applied": bool(emotion.get("asr_calibration_applied")),
        "reason": emotion.get("asr_calibration_reason", ""),
        "label_before": label_before_calibration,
        "label_after": label_after_calibration,
    }
    trace["arbitration"] = {
        "source": emotion.get("arbitration_source", ""),
        "reason": emotion.get("arbitration_reason", ""),
        "model_label": emotion.get("model_emotion_label"),
        "final_label": emotion.get("final_emotion_label"),
    }
    trace["asr"] = {
        "provider": asr_result.get("provider", ""),
        "text": asr_result.get("text", ""),
        "confidence": asr_result.get("confidence", 0.0),
        "error": asr_result.get("error", ""),
    }
    trace["text_merge"] = {
        "user_input": text or "",
        "asr_text": asr_result.get("text", "") or "",
        "merged_text": merged_text,
        "source": text_source,
    }
    mods = trace.get("modalities")
    if isinstance(mods, dict):
        video_byte_count = int((metadata or {}).get("video_byte_count") or 0) or len(video_chunk_b64 or "")
        audio_byte_count = int((metadata or {}).get("audio_byte_count") or 0) or len(audio_chunk_b64 or "")
        if not isinstance(mods.get("video"), dict):
            mods["video"] = {
                "received_bytes": video_byte_count,
                "preprocessed": video_byte_count > 0,
                "decode_mode": mods.get("video_decode", "unknown"),
            }
        elif not mods["video"].get("received_bytes"):
            mods["video"]["received_bytes"] = video_byte_count
        if not isinstance(mods.get("audio"), dict):
            mods["audio"] = {
                "received_bytes": audio_byte_count,
                "preprocessed": audio_byte_count > 0,
                "temporal_windows": mods.get("audio_windows"),
            }
        elif not mods["audio"].get("received_bytes"):
            mods["audio"]["received_bytes"] = audio_byte_count
        if isinstance(mods.get("text"), dict):
            mods["text"]["source"] = text_source
            mods["text"]["merged_text"] = merged_text
        else:
            mods["text"] = {
                "source": text_source,
                "merged_text": merged_text,
                "char_len": len(merged_text),
                "preprocessed": bool(merged_text.strip()),
            }
        trace["modalities"] = mods
    emotion["pipeline_trace"] = trace
    logger.info(
        "[emotion_infer] result session=%s source=%s provider=%s label=%s emotion_id=%s "
        "conf=%.3f ms=%.1f preset=%s fusion=%s ckpt=%s degraded=%s",
        session_id,
        emotion.get("inference_source"),
        emotion.get("model_provider"),
        emotion.get("emotion_label"),
        emotion.get("emotion_id"),
        float(emotion.get("confidence") or 0),
        float(emotion.get("inference_ms") or 0),
        emotion.get("checkpoint_preset"),
        emotion.get("fusion_strategy"),
        emotion.get("checkpoint_file"),
        emotion.get("degraded_mode"),
    )
    if not str(emotion.get("inference_source", "")).startswith("checkpoint"):
        logger.warning(
            "[emotion_infer] NOT using trained checkpoint for session=%s (source=%s provider=%s)",
            session_id,
            emotion.get("inference_source"),
            emotion.get("model_provider"),
        )
    session_store.add(
        session_id,
        "emotion",
        {"emotion": emotion, "asr": asr_result},
    )
    await _broadcast(session_id, {"event": "emotion", "emotion": emotion, "asr": asr_result})
    emotion["asr_text"] = asr_result.get("text", "")
    emotion["asr_confidence"] = float(asr_result.get("confidence", 0.0) or 0.0)
    emotion["asr_error"] = str(asr_result.get("error", "") or "")
    emotion["asr_provider"] = str(asr_result.get("provider", "") or "")
    probs = emotion.get("all_probs") or []
    names = ["happy", "sad", "angry", "fear", "neutral", "anxious", "other"]
    if probs:
        indexed = sorted(enumerate(probs), key=lambda x: -x[1])[:3]
        emotion["top_emotions"] = [
            {"label": names[i], "prob": float(p)} for i, p in indexed if i < len(names)
        ]
    return EmotionInferResponse(**emotion)


@router.post("/emotion/infer", response_model=EmotionInferResponse)
async def emotion_infer(payload: EmotionInferRequest) -> EmotionInferResponse:
    return await _emotion_infer_core(
        session_id=payload.session_id,
        text=payload.text,
        video_chunk_b64=payload.video_chunk_b64,
        audio_chunk_b64=payload.audio_chunk_b64,
        metadata=payload.metadata,
    )


@router.post("/emotion/infer-upload", response_model=EmotionInferResponse)
async def emotion_infer_upload(
    session_id: str = Form(...),
    text: str = Form(""),
    video: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    metadata: str = Form("{}"),
) -> EmotionInferResponse:
    """Binary multipart upload — avoids large JSON/base64 bodies (friendlier for dev port-forward)."""
    video_b64 = ""
    audio_b64 = ""
    video_byte_count = 0
    audio_byte_count = 0
    audio_raw: bytes = b""
    video_mime = ""
    video_filename = ""
    if video is not None:
        raw = await video.read()
        video_byte_count = len(raw)
        video_mime = video.content_type or ""
        video_filename = video.filename or ""
        if raw:
            video_b64 = base64.b64encode(raw).decode("ascii")
    if audio is not None:
        audio_raw = await audio.read()
        audio_byte_count = len(audio_raw)
        if audio_raw:
            audio_b64 = base64.b64encode(audio_raw).decode("ascii")
    meta: Dict[str, Any] = {}
    if metadata:
        try:
            parsed = json.loads(metadata)
            if isinstance(parsed, dict):
                meta = parsed
        except Exception:
            pass
    if video_mime:
        meta["video_mime"] = video_mime
    if video_filename:
        meta["video_filename"] = video_filename
    meta.update(
        {
            "upload": "multipart",
            "video_byte_count": video_byte_count,
            "audio_byte_count": audio_byte_count,
            "audio_raw": audio_raw,
        }
    )
    logger.info(
        "emotion_infer_upload session=%s video_bytes=%d audio_bytes=%d",
        session_id,
        video_byte_count,
        audio_byte_count,
    )
    return await _emotion_infer_core(
        session_id=session_id,
        text=text,
        video_chunk_b64=video_b64,
        audio_chunk_b64=audio_b64,
        metadata=meta,
    )


@router.post("/emotion/upload-chunk")
async def emotion_upload_chunk(
    session_id: str = Form(...),
    field: str = Form(...),
    chunk_index: int = Form(...),
    total_chunks: int = Form(...),
    chunk: UploadFile = File(...),
) -> dict:
    data = await chunk.read()
    try:
        put_chunk(session_id, field, chunk_index, total_chunks, data)
    except ValueError as exc:
        return {"accepted": False, "error": str(exc)}
    return {
        "session_id": session_id,
        "field": field,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "accepted": True,
        "chunk_bytes": len(data),
    }


@router.post("/emotion/upload-chunks-batch")
async def emotion_upload_chunks_batch(
    session_id: str = Form(...),
    field: str = Form(...),
    start_index: int = Form(...),
    total_chunks: int = Form(...),
    chunks: List[UploadFile] = File(...),
) -> dict:
    """一次 HTTP 请求写入多段分块，减少 Cursor 端口转发下的请求次数。"""
    parts = []
    for offset, upload in enumerate(chunks):
        chunk_index = start_index + offset
        if chunk_index >= total_chunks:
            break
        data = await upload.read()
        try:
            put_chunk(session_id, field, chunk_index, total_chunks, data)
        except ValueError as exc:
            return {"accepted": False, "error": str(exc), "failed_index": chunk_index}
        parts.append({"chunk_index": chunk_index, "chunk_bytes": len(data)})
    return {
        "session_id": session_id,
        "field": field,
        "start_index": start_index,
        "accepted": True,
        "count": len(parts),
        "parts": parts,
    }


@router.post("/emotion/infer-from-upload", response_model=EmotionInferResponse)
async def emotion_infer_from_upload(
    session_id: str = Form(...),
    text: str = Form(""),
    metadata: str = Form(""),
) -> EmotionInferResponse:
    try:
        audio_raw, video_raw = assemble_upload(session_id)
    except ValueError as exc:
        logger.warning("infer-from-upload assemble failed session=%s: %s", session_id, exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    extra: Dict[str, Any] = {}
    if metadata:
        try:
            extra = json.loads(metadata)
        except json.JSONDecodeError:
            extra = {"metadata_raw": metadata}
    video_b64 = base64.b64encode(video_raw).decode("ascii") if video_raw else ""
    audio_b64 = base64.b64encode(audio_raw).decode("ascii") if audio_raw else ""
    try:
        return await _emotion_infer_core(
            session_id=session_id,
            text=text,
            video_chunk_b64=video_b64,
            audio_chunk_b64=audio_b64,
            metadata={
                **extra,
                "upload": "chunked",
                "video_byte_count": len(video_raw),
                "audio_byte_count": len(audio_raw),
                "audio_raw": audio_raw,
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("infer-from-upload failed session=%s: %s", session_id, exc)
        raise HTTPException(status_code=500, detail=f"推理失败: {exc}") from exc


@router.post("/agent/respond", response_model=AgentRespondResponse)
async def agent_respond(payload: AgentRespondRequest) -> AgentRespondResponse:
    response = llm_service.generate_response(
        emotion_label=payload.emotion_label,
        confidence=payload.confidence,
        context_text=payload.context_text,
        all_probs=payload.all_probs,
        all_probs_labeled=payload.all_probs_labeled,
        valence=payload.valence,
        arousal=payload.arousal,
        top_emotions=payload.top_emotions,
    )
    session_store.add(payload.session_id, "agent", response)
    await _broadcast(payload.session_id, {"event": "agent", "response": response})
    return AgentRespondResponse(**response)


@router.get("/session/{session_id}/events")
def session_events(session_id: str) -> List[dict]:
    events = session_store.list_events(session_id)
    return [e.model_dump() for e in events]


@router.websocket("/session/{session_id}/stream")
async def session_stream(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    ws_clients[session_id].append(websocket)
    try:
        await websocket.send_text(json.dumps({"event": "connected", "session_id": session_id}, ensure_ascii=False))
        while True:
            # keepalive/read incoming messages for future client control commands.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        ws_clients[session_id] = [ws for ws in ws_clients[session_id] if ws is not websocket]
