"""Fuse audio + text emotion channels; flag incongruence (emotion gap)."""

_SEVERITY = {
    "high_distress": 0,
    "distressed": 1,
    "emotional_mismatch": 2,
    "agitated": 3,
    "low_mood": 4,
    "sad": 5,
    "anxious": 5,
    "angry": 6,
    "neutral": 8,
    "happy": 9,
}


def _sev(label: str) -> int:
    return _SEVERITY.get(label, 7)


def _base_risk(audio: dict, text_label: str) -> float:
    stress = float(audio.get("stress_score") or 0)
    energy = float(audio.get("energy") or 0)
    a_label = audio.get("label") or "neutral"

    risk = stress * 0.45 + (1.0 - energy) * 0.2
    if a_label in ("distressed", "agitated", "low_mood"):
        risk += 0.15
    if text_label in ("sad", "anxious", "angry"):
        risk += 0.12
    return max(0.0, min(1.0, risk))


def combine(audio_em: dict, text_em: dict) -> dict:
    audio = audio_em or {}
    text_label = (text_em or {}).get("label") or "neutral"
    a_label = audio.get("label") or "neutral"

    mismatch = False
    if text_label == "happy" and a_label in ("distressed", "low_mood", "agitated"):
        mismatch = True
    if text_label in ("sad", "anxious", "angry") and a_label == "happy":
        mismatch = True
    if {text_label, a_label} == {"happy", "sad"} or {text_label, a_label} == {"happy", "anxious"}:
        mismatch = True

    if mismatch:
        final = "emotional_mismatch"
        reason = (
            "Words and voice cues diverge — possible emotional incongruence "
            f"(text: {text_label}, voice: {a_label})."
        )
        combined_risk = min(1.0, _base_risk(audio, text_label) + 0.22)
    else:
        candidates = [text_label, a_label]
        final = min(candidates, key=_sev)
        if final == "neutral" and text_label != "neutral":
            final = text_label
        elif final == "neutral" and a_label != "neutral":
            final = a_label
        reason = f"Aligned read: dominant tone is {final.replace('_', ' ')}."
        combined_risk = _base_risk(audio, text_label)

    if combined_risk >= 0.72 and final not in ("emotional_mismatch", "happy"):
        final = "high_distress"
        reason = "Elevated distress signal from combined voice and language cues."

    if final in ("distressed", "agitated") and combined_risk < 0.55:
        combined_risk = 0.55
    if final == "high_distress":
        combined_risk = min(1.0, max(combined_risk, 0.75))

    return {
        "final_emotion": final,
        "combined_risk": round(float(combined_risk), 3),
        "reason": reason,
        "audio_label": a_label,
        "text_label": text_label,
        "mismatch": mismatch,
    }
