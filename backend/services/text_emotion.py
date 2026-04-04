"""Text emotion: keyword cues + optional VADER sentiment (better than keywords alone)."""

from __future__ import annotations

import re

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    _vader = SentimentIntensityAnalyzer()
except Exception:
    _vader = None

_POS = re.compile(
    r"\b(happy|happiness|glad|joy|joyful|excited|exciting|great|love|loving|wonderful|amazing|"
    r"thanks|thank you|good|better|awesome|fantastic|blessed|grateful|proud|relieved|calm|peaceful)\b",
    re.I,
)
_NEG = re.compile(
    r"\b(sad|sadness|depressed|depression|hopeless|hopelessness|cry|crying|tired|exhausted|empty|lonely|"
    r"anxious|anxiety|worried|worry|scared|afraid|fear|panic|nervous|stressed|stress|overwhelmed|"
    r"angry|anger|mad|furious|rage|hate|annoyed|frustrated|frustration|hurt|pain|terrible|awful|worst|"
    r"disgusted|ashamed|guilty|numb|broken|giving up|cannot sleep|can't sleep|cant sleep|no sleep|not sleeping|"
    r"insomnia|shaking|heart racing|racing thoughts)\b",
    re.I,
)
_ANG = re.compile(
    r"\b(angry|anger|furious|rage|mad|hate|annoyed|frustrated|pissed|irritated|livid)\b",
    re.I,
)
_ANX = re.compile(
    r"\b(anxious|anxiety|worried|worry|nervous|panic|stress|stressed|scared|afraid|fear|overwhelmed|shaking|"
    r"can't cope|cannot cope)\b",
    re.I,
)
_MASK = re.compile(
    r"\b(i'?m fine|i am fine|everything is fine|all good|nothing wrong|i'?m ok|just tired|don'?t worry about me)\b",
    re.I,
)


def detect(text: str) -> dict:
    t = (text or "").strip()
    if not t:
        return {"label": "neutral", "confidence": 0.0}

    pos = len(_POS.findall(t))
    neg = len(_NEG.findall(t))
    ang = len(_ANG.findall(t))
    anx = len(_ANX.findall(t))
    mask = len(_MASK.findall(t))

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

    confidence = min(1.0, 0.38 + 0.12 * (pos + neg + ang + anx))

    if _vader is not None:
        vs = _vader.polarity_scores(t)
        compound = float(vs["compound"])
        if label == "neutral":
            if compound <= -0.45:
                label = "sad"
                confidence = max(confidence, 0.55)
            elif compound <= -0.2:
                label = "low_mood"
                confidence = max(confidence, 0.48)
            elif compound >= 0.42:
                label = "happy"
                confidence = max(confidence, 0.52)
        if mask >= 1 and compound < -0.15 and label in ("neutral", "happy"):
            label = "emotional_mismatch"
            confidence = max(confidence, 0.5)
        if label == "happy" and compound < -0.25:
            label = "emotional_mismatch"
            confidence = max(confidence, 0.55)

    return {"label": label, "confidence": round(confidence, 3)}
