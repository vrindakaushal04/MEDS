"""Empathetic reply via Oumi fine-tuned model (OpenAI-compatible server)."""

from __future__ import annotations

import json
from typing import Optional
import urllib.error
import urllib.request

from config import OUMI_API_KEY, OUMI_BASE_URL, OUMI_MODEL, USE_LLM


_SYSTEM = """You are MEDS — a calm, supportive voice-wellbeing assistant.
The user message includes detected emotions from text and audio fusion.
Reply in one or two short sentences (max 35 words). Be warm, non-clinical, and practical.
Do not diagnose. If distress seems high, gently suggest grounding or reaching out to someone they trust."""


def _fallback_reply(text: str, analysis: dict) -> str:
    final = (analysis or {}).get("final_emotion", "neutral")
    risk = float((analysis or {}).get("combined_risk") or 0)
    if final == "emotional_mismatch":
        return (
            "It sounds like part of you is trying to sound okay while another part feels heavier. "
            "That is more common than you think — what feels most true for you right now?"
        )
    if risk >= 0.65 or final in ("distressed", "agitated", "high_distress"):
        return (
            "I hear that this is a lot. Try one slow breath with me, then name one small next step. "
            "If you are unsafe, please contact local emergency services or a crisis line."
        )
    if final in ("sad", "anxious", "low_mood"):
        return "Thank you for sharing that. You are not alone in feeling this way. What would feel slightly easier in the next hour?"
    if final == "happy":
        return "I am glad to sense some lightness in what you shared. What is contributing to that for you?"
    return "I am here with you. Say more about what is on your mind, in your own words."


def _chat_oumi(user_content: str) -> Optional[str]:
    url = f"{OUMI_BASE_URL}/chat/completions"
    body = json.dumps(
        {
            "model": OUMI_MODEL,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.65,
            "max_tokens": 180,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OUMI_API_KEY}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        choices = data.get("choices") or []
        if not choices:
            return None
        msg = choices[0].get("message") or {}
        content = (msg.get("content") or "").strip()
        return content or None
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, TimeoutError):
        return None


def generate(text: str, analysis: dict) -> str:
    fused = analysis or {}
    payload = (
        f"User said (transcript or typed): {text}\n"
        f"Fusion: final_emotion={fused.get('final_emotion')}, "
        f"combined_risk={fused.get('combined_risk')}, "
        f"mismatch={fused.get('mismatch')}, "
        f"reason={fused.get('reason')}"
    )

    if USE_LLM:
        reply = _chat_oumi(payload)
        if reply:
            return reply

    return _fallback_reply(text, fused)
