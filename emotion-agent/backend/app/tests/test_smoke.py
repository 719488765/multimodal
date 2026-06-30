from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health() -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json().get("ok") is True


def test_infer_and_agent_flow() -> None:
    session_id = "test_session"
    infer_response = client.post(
        "/api/v1/emotion/infer",
        json={
            "session_id": session_id,
            "text": "我今天有点紧张",
            "video_chunk_b64": "video",
            "audio_chunk_b64": "audio",
            "metadata": {},
        },
    )
    assert infer_response.status_code == 200
    infer_payload = infer_response.json()
    assert "emotion_label" in infer_payload

    reply_response = client.post(
        "/api/v1/agent/respond",
        json={
            "session_id": session_id,
            "emotion_label": infer_payload["emotion_label"],
            "confidence": infer_payload["confidence"],
            "context_text": "测试上下文",
        },
    )
    assert reply_response.status_code == 200
    assert "reply_text" in reply_response.json()
