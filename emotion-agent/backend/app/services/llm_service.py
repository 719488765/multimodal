from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import requests

from app.core.config import settings

logger = logging.getLogger(__name__)

EMOTION_LABEL_ZH = {
    "happy": "开心",
    "sad": "难过",
    "angry": "生气",
    "fear": "害怕",
    "neutral": "平静",
    "anxious": "焦虑",
    "other": "其他",
}


class LLMService:
    def probe(self) -> dict:
        provider = settings.llm_provider.lower().strip()
        if provider in ("template", "mock", ""):
            return {
                "provider": provider or "template",
                "ok": False,
                "message": "LLM_PROVIDER 未配置为 ollama/openai，将使用模板话术",
            }
        if provider == "ollama":
            base = (settings.llm_api_base or "http://127.0.0.1:11434").rstrip("/")
            try:
                resp = requests.get(f"{base}/api/tags", timeout=5)
                resp.raise_for_status()
                models = [m.get("name", "") for m in resp.json().get("models", [])]
                want = settings.llm_model or "qwen2.5:7b-instruct"
                has_model = any(want.split(":")[0] in name for name in models)
                if not models:
                    return {
                        "provider": "ollama",
                        "ok": False,
                        "message": "Ollama 已启动但未拉取模型",
                        "hint": f"请运行: ollama pull {want}",
                        "models": models,
                    }
                if not has_model:
                    return {
                        "provider": "ollama",
                        "ok": False,
                        "message": f"未找到模型 {want}",
                        "hint": f"请运行: ollama pull {want}",
                        "models": models,
                    }
                return {"provider": "ollama", "ok": True, "message": f"model {want} ready", "models": models}
            except Exception as exc:
                return {
                    "provider": "ollama",
                    "ok": False,
                    "message": f"无法连接 Ollama ({base}): {exc}",
                    "hint": "请运行: ollama serve  或  emotion-agent/scripts/start_ollama.sh",
                }
        if provider == "openai":
            base = (settings.llm_api_base or "").strip()
            if not base:
                return {"provider": "openai", "ok": False, "message": "LLM_API_BASE 未配置"}
            return {"provider": "openai", "ok": True, "message": f"API base configured: {base}"}
        return {"provider": provider, "ok": False, "message": f"未知 LLM_PROVIDER={provider}"}

    @staticmethod
    def _format_probs_for_prompt(
        all_probs_labeled: Optional[List[Dict[str, Any]]] = None,
        all_probs: Optional[List[float]] = None,
    ) -> str:
        items: List[Dict[str, Any]] = []
        if all_probs_labeled:
            items = list(all_probs_labeled)
        elif all_probs:
            items = [
                {
                    "label": name,
                    "label_cn": EMOTION_LABEL_ZH.get(name, name),
                    "prob": float(all_probs[i]) if i < len(all_probs) else 0.0,
                }
                for i, name in enumerate(["happy", "sad", "angry", "fear", "neutral", "anxious", "other"])
            ]
        if not items:
            return "（无概率分布）"
        sorted_items = sorted(items, key=lambda x: -float(x.get("prob") or 0))
        return "，".join(
            f"{x.get('label_cn') or EMOTION_LABEL_ZH.get(x.get('label', ''), x.get('label', ''))}"
            f"({x.get('label', '')})={float(x.get('prob') or 0):.3f}"
            for x in sorted_items
        )

    def _template_response(
        self,
        emotion_label: str,
        confidence: float,
        context_text: str = "",
        llm_error: str = "",
        all_probs_labeled: Optional[List[Dict[str, Any]]] = None,
        all_probs: Optional[List[float]] = None,
        valence: Optional[float] = None,
        arousal: Optional[float] = None,
    ) -> dict:
        label_zh = EMOTION_LABEL_ZH.get(emotion_label, emotion_label)
        ctx = (context_text or "").strip()
        ctx_snip = f"我听到你说：「{ctx[:40]}{'…' if len(ctx) > 40 else ''}」。" if ctx else ""
        probs_txt = self._format_probs_for_prompt(all_probs_labeled, all_probs)

        if confidence < 0.45:
            reply = (
                f"{ctx_snip}多模态模型判断你更接近「{label_zh}」（置信度 {confidence:.2f}，分布：{probs_txt}）。"
                "能再用一两句话描述你现在的感受吗？"
            ).strip()
            return {
                "reply_text": reply,
                "tone": "calm",
                "safe_mode": True,
                "llm_provider": "template",
                "llm_error": llm_error or "LLM 不可用，已使用模板兜底",
            }

        template_map = {
            "happy": "从模型判断你状态不错，继续保持这个节奏。",
            "sad": "模型感受到你可能有些低落，先慢一点，深呼吸一下。",
            "angry": "模型感受到你有些激动，先让自己缓一缓。",
            "fear": "模型感受到你有些紧张，我们一步一步来。",
            "neutral": "模型判断你目前比较平静，有需要随时告诉我。",
            "anxious": "模型感受到你有些焦虑，先把注意力放回当下。",
            "other": "我在这里，愿意继续听你说。",
        }
        base = template_map.get(emotion_label, template_map["other"])
        dim_hint = ""
        if valence is not None and arousal is not None:
            dim_hint = f"（效价 {valence:.2f}，唤醒 {arousal:.2f}）"
        reply = f"{ctx_snip}{base}{dim_hint}".strip() if ctx_snip else f"{base}{dim_hint}".strip()
        return {
            "reply_text": reply,
            "tone": "encourage" if emotion_label == "happy" else "comfort",
            "safe_mode": False,
            "llm_provider": "template",
            "llm_error": llm_error or "",
        }

    def _build_messages(
        self,
        emotion_label: str,
        confidence: float,
        context_text: str,
        *,
        all_probs_labeled: Optional[List[Dict[str, Any]]] = None,
        all_probs: Optional[List[float]] = None,
        valence: Optional[float] = None,
        arousal: Optional[float] = None,
    ) -> list:
        system_prompt = (
            "你是一个情绪支持助手。请使用中文，语气温和、简洁。"
            "禁止医疗诊断、禁止极端建议、禁止夸大。"
            "回复必须以多模态情绪模型的分类结果为主；ASR 转写仅作参考。"
            "若 ASR 与模型 top1 情绪不一致，请温和 bridging（例如：听你说…，但从表达看更像是…）。"
            "请给出1-2句可执行建议，长度控制在80字以内。"
        )
        label_zh = EMOTION_LABEL_ZH.get(emotion_label, emotion_label)
        probs_txt = self._format_probs_for_prompt(all_probs_labeled, all_probs)
        user_prompt = (
            f"【情绪模型输出 - 以此为准】\n"
            f"Top1: {emotion_label}（{label_zh}），置信度 {confidence:.3f}\n"
            f"七类概率: {probs_txt}\n"
            f"Valence: {valence if valence is not None else 'N/A'}，"
            f"Arousal: {arousal if arousal is not None else 'N/A'}\n"
            f"【ASR/用户文本 - 仅供参考】\n{(context_text or '').strip() or '（无）'}\n"
            "请生成回复，不要编造用户未说过的事实。"
        )
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    def _call_openai_compatible(
        self,
        emotion_label: str,
        confidence: float,
        context_text: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        base = (settings.llm_api_base or "").rstrip("/")
        if not base:
            raise RuntimeError("LLM_API_BASE is empty")
        model = settings.llm_model or "gpt-4o-mini"
        url = f"{base}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if settings.llm_api_key:
            headers["Authorization"] = f"Bearer {settings.llm_api_key}"
        payload = {
            "model": model,
            "messages": self._build_messages(emotion_label, confidence, context_text, **kwargs),
            "temperature": float(settings.llm_temperature),
            "max_tokens": 180,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=float(settings.llm_timeout_sec))
        resp.raise_for_status()
        data = resp.json()
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        if not content:
            raise RuntimeError("LLM returned empty content")
        return {
            "reply_text": content,
            "tone": "comfort",
            "safe_mode": confidence < 0.45,
            "llm_provider": "openai",
            "llm_model": model,
            "llm_error": "",
        }

    def _call_ollama(
        self,
        emotion_label: str,
        confidence: float,
        context_text: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        base = (settings.llm_api_base or "http://127.0.0.1:11434").rstrip("/")
        model = settings.llm_model or "qwen2.5:7b-instruct"
        url = f"{base}/api/chat"
        payload = {
            "model": model,
            "messages": self._build_messages(emotion_label, confidence, context_text, **kwargs),
            "stream": False,
            "options": {
                "temperature": float(settings.llm_temperature),
            },
        }
        resp = requests.post(url, json=payload, timeout=float(settings.llm_timeout_sec))
        resp.raise_for_status()
        data = resp.json()
        content = (data.get("message", {}) or {}).get("content", "").strip()
        if not content:
            raise RuntimeError("Ollama returned empty content")
        return {
            "reply_text": content,
            "tone": "comfort",
            "safe_mode": confidence < 0.45,
            "llm_provider": "ollama",
            "llm_model": model,
            "llm_error": "",
        }

    def generate_response(
        self,
        emotion_label: str,
        confidence: float,
        context_text: str,
        *,
        all_probs: Optional[List[float]] = None,
        all_probs_labeled: Optional[List[Dict[str, Any]]] = None,
        valence: Optional[float] = None,
        arousal: Optional[float] = None,
        top_emotions: Optional[List[Dict[str, Any]]] = None,
    ) -> dict:
        kwargs = {
            "all_probs": all_probs,
            "all_probs_labeled": all_probs_labeled or top_emotions,
            "valence": valence,
            "arousal": arousal,
        }
        provider = settings.llm_provider.lower().strip()
        if provider == "openai":
            try:
                return self._call_openai_compatible(emotion_label, confidence, context_text, **kwargs)
            except Exception as exc:
                logger.warning("[LLM] openai failed, template fallback: %s", exc)
                return self._template_response(
                    emotion_label, confidence, context_text, llm_error=str(exc), **kwargs
                )
        if provider == "ollama":
            try:
                return self._call_ollama(emotion_label, confidence, context_text, **kwargs)
            except Exception as exc:
                logger.warning("[LLM] ollama failed, template fallback: %s", exc)
                return self._template_response(
                    emotion_label, confidence, context_text, llm_error=str(exc), **kwargs
                )
        return self._template_response(
            emotion_label,
            confidence,
            context_text,
            llm_error=f"LLM_PROVIDER={provider} 未启用生成式模型",
            **kwargs,
        )
