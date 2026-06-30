from __future__ import annotations

import base64
import logging
import time
from typing import Optional, Tuple

import requests

from app.core.config import settings

logger = logging.getLogger(__name__)


class ASRService:
    """ASR：对接 asr-local（faster-whisper）；mock 模式不返回虚假固定台词。"""

    def decode_payload(self, audio_chunk_b64: str) -> Tuple[bytes, str]:
        if not audio_chunk_b64:
            return b"", "wav"
        if "," in audio_chunk_b64 and audio_chunk_b64.strip().startswith("data:"):
            header, payload = audio_chunk_b64.split(",", 1)
            ext = "wav"
            if "audio/webm" in header:
                ext = "webm"
            elif "audio/wav" in header:
                ext = "wav"
            elif "audio/mp3" in header:
                ext = "mp3"
            return base64.b64decode(payload), ext
        return base64.b64decode(audio_chunk_b64), "wav"

    def probe(self) -> dict:
        """检查当前 ASR 配置是否可用（供 /health 与启动日志）。"""
        provider = settings.asr_provider.lower().strip()
        if provider == "mock":
            return {
                "provider": "mock",
                "ok": False,
                "message": "ASR_PROVIDER=mock 不会真实转写；请改为 whisper_api 并启动 asr-local",
            }
        if provider in ("whisper_api", "auto"):
            url = (settings.asr_whisper_api_url or "").strip()
            if not url:
                return {"provider": provider, "ok": False, "message": "ASR_WHISPER_API_URL 未配置"}
            health_url = url.replace("/v1/audio/transcriptions", "/health")
            try:
                resp = requests.get(health_url, timeout=3)
                if resp.status_code == 200:
                    return {"provider": "whisper_api", "ok": True, "message": "asr-local reachable", "health": resp.json()}
                return {
                    "provider": "whisper_api",
                    "ok": False,
                    "message": f"ASR health HTTP {resp.status_code}",
                    "health_url": health_url,
                }
            except Exception as exc:
                return {
                    "provider": "whisper_api",
                    "ok": False,
                    "message": f"无法连接 ASR 服务: {exc}",
                    "health_url": health_url,
                    "hint": "请先运行: cd emotion-agent/asr-local && ./start_server.sh",
                }
        if provider == "whisper_local":
            try:
                import whisper  # type: ignore  # noqa: F401
                return {"provider": "whisper_local", "ok": True, "message": "openai-whisper import ok"}
            except Exception as exc:
                return {"provider": "whisper_local", "ok": False, "message": f"whisper 未安装: {exc}"}
        return {"provider": provider, "ok": False, "message": f"未知 ASR_PROVIDER={provider}"}

    def _transcribe_whisper_api_bytes(self, raw: bytes, ext: str) -> dict:
        if not settings.asr_whisper_api_url:
            return {
                "text": "",
                "confidence": 0.0,
                "provider": "whisper_api",
                "error": "ASR_WHISPER_API_URL 为空",
            }
        if not raw:
            return {"text": "", "confidence": 0.0, "provider": "whisper_api"}
        files = {"file": (f"capture.{ext}", raw, "application/octet-stream")}
        data = {"model": settings.asr_whisper_api_model}
        if settings.asr_whisper_api_language:
            data["language"] = settings.asr_whisper_api_language
        headers = {}
        if settings.asr_whisper_api_key:
            headers["Authorization"] = f"Bearer {settings.asr_whisper_api_key}"
        last_err: Optional[Exception] = None
        t0 = time.time()
        read_timeout = float(settings.asr_timeout_sec)
        connect_timeout = min(5.0, read_timeout)
        max_attempts = max(1, int(getattr(settings, "asr_max_retries", 1)))
        for attempt in range(max_attempts):
            try:
                resp = requests.post(
                    settings.asr_whisper_api_url,
                    headers=headers,
                    data=data,
                    files=files,
                    timeout=(connect_timeout, read_timeout),
                )
                resp.raise_for_status()
                result = resp.json()
                text = (result.get("text") or "").strip()
                latency_ms = int((time.time() - t0) * 1000)
                logger.info(
                    "[ASR] whisper_api ok text_len=%d latency_ms=%d model=%s",
                    len(text),
                    latency_ms,
                    result.get("model", settings.asr_whisper_api_model),
                )
                return {
                    "text": text,
                    "confidence": 0.9 if text else 0.0,
                    "provider": "whisper_api",
                    "latency_ms": latency_ms,
                    "model": result.get("model", settings.asr_whisper_api_model),
                }
            except Exception as e:
                last_err = e
                logger.warning("[ASR] whisper_api attempt %d/%d failed: %s", attempt + 1, max_attempts, e)
                if attempt + 1 < max_attempts:
                    time.sleep(0.2)
                    continue
        return {
            "text": "",
            "confidence": 0.0,
            "provider": "whisper_api",
            "error": str(last_err),
            "hint": "请确认 asr-local 已启动: ./emotion-agent/asr-local/start_server.sh",
        }

    def _transcribe_whisper_local_bytes(self, raw: bytes, ext: str) -> dict:
        import tempfile
        from pathlib import Path

        if not raw:
            return {"text": "", "confidence": 0.0, "provider": "whisper_local"}
        try:
            import whisper  # type: ignore
        except Exception as e:
            return {
                "text": "",
                "confidence": 0.0,
                "provider": "whisper_local",
                "error": f"whisper import failed: {e}",
            }

        tmp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as f:
                f.write(raw)
                tmp_path = Path(f.name)
            model = whisper.load_model(settings.asr_whisper_local_model)
            result = model.transcribe(str(tmp_path), language=settings.asr_whisper_api_language or "zh")
            text = (result.get("text") or "").strip()
            return {"text": text, "confidence": 0.85 if text else 0.0, "provider": "whisper_local"}
        except Exception as e:
            return {"text": "", "confidence": 0.0, "provider": "whisper_local", "error": str(e)}
        finally:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def transcribe_bytes(self, raw: bytes, ext: str = "wav") -> dict:
        provider = settings.asr_provider.lower().strip()

        if provider == "mock":
            return {
                "text": "",
                "confidence": 0.0,
                "provider": "mock",
                "error": "ASR 处于 mock 模式，未执行真实转写。请设置 ASR_PROVIDER=whisper_api 并启动 asr-local。",
            }

        if provider == "auto":
            return self._transcribe_whisper_api_bytes(raw, ext)

        if provider == "whisper_api":
            return self._transcribe_whisper_api_bytes(raw, ext)

        if provider == "whisper_local":
            return self._transcribe_whisper_local_bytes(raw, ext)

        return {
            "text": "",
            "confidence": 0.0,
            "provider": provider,
            "error": f"未知 ASR_PROVIDER={provider}",
        }

    def transcribe(self, audio_chunk_b64: str) -> dict:
        raw, ext = self.decode_payload(audio_chunk_b64)
        return self.transcribe_bytes(raw, ext)
