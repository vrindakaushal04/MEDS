"""Fuse audio + text: confident prosody can override neutral/happy words (hidden affect)."""

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

_PROSODY_STRONG = frozenset(
    {"anxious", "sad", "distressed", "low_mood", "agitated", "angry", "happy"}
)


def _sev(label: str) -> int:
    return _SEVERITY.get(label, 7)


def _base_risk(audio: dict, text_label: str) -> float:
    stress = float(audio.get("stress_score") or 0)
    energy = float(audio.get("energy") or 0)
    a_label = audio.get("label") or "neutral"
    pconf = float(audio.get("prosody_confidence") or 0.5)
    arousal = float(audio.get("arousal") or 0.0)

    risk = stress * 0.42 + (1.0 - energy) * 0.18 + arousal * 0.12
    risk += 0.08 * pconf * (1.0 if a_label in ("distressed", "anxious", "sad", "low_mood") else 0.5)

    if a_label in ("distressed", "agitated", "low_mood"):
        risk += 0.14
    if text_label in ("sad", "anxious", "angry", "emotional_mismatch", "low_mood", "distressed"):
        risk += 0.12
    return max(0.0, min(1.0, risk))


def _voice_led_hidden_affect(text_label: str, a_label: str, audio: dict) -> bool:
    """Trust timbre / pitch / energy over neutral or upbeat words when prosody is confident."""
    if a_label == "neutral":
        return False
    pconf = float(audio.get("prosody_confidence") or 0)
    valence = float(audio.get("valence") or 0.0)
    # Lowered threshold from 0.42 → 0.35, real mic speech may have slightly lower confidence
    if pconf < 0.35:
        return False

    if text_label == "neutral" and a_label in _PROSODY_STRONG:
        return True

    if text_label == "happy" and a_label in ("sad", "anxious", "distressed", "low_mood"):
        return pconf >= 0.35

    if text_label == "happy" and a_label in ("agitated", "angry"):
        return pconf >= 0.45

    if text_label == "happy" and valence < -0.2 and a_label not in ("happy", "neutral"):
        return pconf >= 0.40

    return False



def _hidden_reason(a_label: str, audio: dict) -> str:
    stress = float(audio.get("stress_score") or 0)
    arousal = float(audio.get("arousal") or 0)
    valence = float(audio.get("valence") or 0)
    bits = []
    if stress > 0.48:
        bits.append("vocal tension / uneven delivery")
    elif stress > 0.35:
        bits.append("subtle vocal strain")
    if arousal > 0.55:
        bits.append("elevated vocal activation")
    if valence < -0.25:
        bits.append("darker timbre / low energy coloring")

    cue = ", ".join(bits[:2]) if bits else "pitch, loudness, and rhythm patterns"
    return (
        f"Literal words are mild, but how it sounds ({cue}) aligns with "
        f"{a_label.replace('_', ' ')} — possible meaning beneath the surface text."
    )


def combine(audio_em: dict, text_em: dict) -> dict:
    audio = audio_em or {}
    text_label = (text_em or {}).get("label") or "neutral"
    a_label = audio.get("label") or "neutral"
    pconf = float(audio.get("prosody_confidence") or 0)

    mismatch = False
    if text_label == "happy" and a_label in ("distressed", "low_mood", "agitated", "sad", "anxious"):
        mismatch = True
    if text_label in ("sad", "anxious", "angry") and a_label == "happy":
        mismatch = True
    if {text_label, a_label} == {"happy", "sad"} or {text_label, a_label} == {"happy", "anxious"}:
        mismatch = True
    if text_label == "emotional_mismatch":
        mismatch = True

    voice_led = _voice_led_hidden_affect(text_label, a_label, audio)

    if mismatch:
        final = "emotional_mismatch"
        if text_label == "emotional_mismatch":
            reason = (
                "Surface wording minimizes feelings, but sentiment cues suggest more underneath; "
                f"voice read: {a_label.replace('_', ' ')}."
            )
        else:
            reason = (
                "Words and voice cues diverge — possible emotional incongruence "
                f"(text: {text_label}, voice: {a_label})."
            )
        combined_risk = min(1.0, _base_risk(audio, text_label) + 0.24)
    elif voice_led:
        final = a_label
        reason = _hidden_reason(a_label, audio)
        combined_risk = min(1.0, _base_risk(audio, text_label) + 0.12 + 0.08 * pconf)
    else:
        candidates = [text_label, a_label]
        final = min(candidates, key=_sev)
        if final == "neutral" and text_label != "neutral":
            final = text_label
            reason = f"Aligned read: dominant tone is {final.replace('_', ' ')}."
            combined_risk = _base_risk(audio, text_label)
        elif final == "neutral" and a_label != "neutral" and pconf >= 0.5:
            final = a_label
            reason = _hidden_reason(a_label, audio)
            combined_risk = _base_risk(audio, text_label)
        else:
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
        "voice_led": voice_led,
        "prosody_confidence": round(pconf, 3),
    }
