"""Lightweight lexical emotion signals (no external ML API)."""

import re

_POS = re.compile(
    r"\b(happy|glad|joy|excited|great|love|wonderful|thanks|thank you|good)\b", re.I
)
_NEG = re.compile(
    r"\b(sad|depressed|hopeless|cry|tired|anxious|worried|scared|afraid|angry|mad|hate|terrible|awful)\b",
    re.I,
)
_ANG = re.compile(r"\b(angry|furious|mad|hate|annoyed|frustrated)\b", re.I)
_ANX = re.compile(r"\b(anxious|worried|nervous|panic|stress|scared|afraid)\b", re.I)


def detect(text: str) -> dict:
    t = (text or "").strip()
    if not t:
        return {"label": "neutral", "confidence": 0.0}

    pos = len(_POS.findall(t))
    neg = len(_NEG.findall(t))
    ang = len(_ANG.findall(t))
    anx = len(_ANX.findall(t))

    if ang >= 1 and ang >= anx:
        label = "angry"
    elif anx >= 1:
        label = "anxious"
    elif neg > pos:
        label = "sad"
    elif pos > neg:
        label = "happy"
    else:
        label = "neutral"

    confidence = min(1.0, 0.35 + 0.15 * (pos + neg + ang + anx))
    return {"label": label, "confidence": round(confidence, 3)}
