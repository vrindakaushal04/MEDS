"""Prosody-first emotion cues from audio (librosa). Not lexical — voice timbre, pitch, energy dynamics."""

from __future__ import annotations

import math
import os
import tempfile
from typing import Dict, List, Tuple

import numpy as np

try:
    import librosa
except ImportError:
    librosa = None  # type: ignore[assignment, misc]

# Clip length cap (pyin is heavier on long files)
_MAX_SEC = 18.0
_HOP = 512
_FRAME = 2048


def _safe_norm(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.5
    return float(max(0.0, min(1.0, (x - lo) / (hi - lo))))


def _sigmoid(x: float, k: float = 8.0, mid: float = 0.5) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-k * (x - mid)))
    except OverflowError:
        return 1.0 if x > mid else 0.0


def _frame_features(y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rms = librosa.feature.rms(y=y, frame_length=_FRAME, hop_length=_HOP)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=_FRAME, hop_length=_HOP)[0]
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=_HOP)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=_HOP)[0]
    return rms, zcr, cent, rolloff


def _f0_stats(y: np.ndarray, sr: int) -> Tuple[float, float, float]:
    """Mean F0 (Hz), std F0 (Hz), voiced fraction — NaNs stripped."""
    fmin = librosa.note_to_hz("C2")
    fmax = librosa.note_to_hz("C7")
    f0, vf, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, hop_length=_HOP)
    voiced = vf.astype(bool) & np.isfinite(f0)
    if not np.any(voiced):
        return 0.0, 0.0, 0.0
    fv = f0[voiced]
    return float(np.nanmean(fv)), float(np.nanstd(fv)), float(np.mean(voiced))


def _onset_proxy(y: np.ndarray, sr: int) -> Tuple[float, float]:
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=_HOP)
    if env.size == 0:
        return 0.0, 0.0
    e = env / (float(np.max(env)) + 1e-9)
    return float(np.mean(e)), float(np.std(e))


def _score_emotions(feat: Dict[str, float]) -> Tuple[str, float, Dict[str, float]]:
    """
    Map continuous prosody features to emotion labels using competing scores.
    Designed so flat / quiet speech is not always 'neutral' when dynamics say otherwise.
    """
    e = feat["energy_mean"]
    e_std = feat["energy_jitter"]
    z = feat["zcr_mean"]
    z_std = feat["zcr_jitter"]
    c = feat["brightness"]
    r = feat["rolloff_norm"]
    f0m = feat["f0_mean_norm"]
    f0s = feat["f0_shake"]
    vf = feat["voiced_frac"]
    on_m = feat["onset_mean"]
    on_s = feat["onset_std"]

    # Voicing too weak — avoid pretending we know tone
    if vf < 0.08 and e < 0.12:
        return "neutral", 0.25, {}

    scores: Dict[str, float] = {}

    # Angry / agitated: high energy, sharp attacks, bright spectrum, often elevated ZCR
    scores["agitated"] = (
        0.32 * _sigmoid(e, k=10, mid=0.42)
        + 0.22 * _sigmoid(z, k=12, mid=0.12)
        + 0.18 * _sigmoid(c, k=10, mid=0.45)
        + 0.18 * _sigmoid(on_m, k=10, mid=0.35)
        + 0.10 * _sigmoid(e_std, k=12, mid=0.22)
    )

    # Anxious: shaky pitch / energy, irregular onsets, high micro-variation
    scores["anxious"] = (
        0.34 * _sigmoid(f0s, k=11, mid=0.18)
        + 0.28 * _sigmoid(e_std, k=11, mid=0.20)
        + 0.18 * _sigmoid(z_std, k=11, mid=0.08)
        + 0.12 * _sigmoid(on_s, k=10, mid=0.12)
        + 0.08 * _sigmoid(1.0 - e, k=8, mid=0.55)
    )

    # Sad / low mood: lower relative energy, flatter pitch contour, darker spectrum
    scores["sad"] = (
        0.30 * _sigmoid(1.0 - e, k=10, mid=0.52)
        + 0.28 * _sigmoid(1.0 - f0s, k=8, mid=0.55)
        + 0.22 * _sigmoid(1.0 - c, k=10, mid=0.48)
        + 0.12 * _sigmoid(1.0 - f0m, k=8, mid=0.48)
        + 0.08 * _sigmoid(1.0 - on_m, k=8, mid=0.55)
    )

    scores["low_mood"] = (
        0.38 * _sigmoid(1.0 - e, k=9, mid=0.45)
        + 0.28 * _sigmoid(1.0 - on_m, k=9, mid=0.50)
        + 0.18 * _sigmoid(1.0 - c, k=8, mid=0.50)
        + 0.16 * _sigmoid(0.25 - abs(f0s - 0.18), k=14, mid=0.0)
    )

    # Happy / animated: lively energy + pitch movement, not chaotic
    scores["happy"] = (
        0.28 * _sigmoid(e, k=9, mid=0.38)
        + 0.26 * _sigmoid(f0s, k=9, mid=0.22)
        + 0.22 * _sigmoid(c, k=8, mid=0.42)
        + 0.14 * _sigmoid(on_m, k=9, mid=0.32)
        + 0.10 * (1.0 - _sigmoid(e_std, k=12, mid=0.35))
    )

    # Distressed: strong negative prosody + instability (cry/shake)
    scores["distressed"] = (
        0.30 * _sigmoid(e_std, k=10, mid=0.24)
        + 0.26 * _sigmoid(f0s, k=10, mid=0.24)
        + 0.22 * _sigmoid(1.0 - e, k=8, mid=0.48)
        + 0.22 * _sigmoid(z_std, k=10, mid=0.09)
    )

    scores["neutral"] = 0.55 * (1.0 - max(scores.values())) + 0.25 * (1.0 - vf) + 0.20 * (1.0 - on_m)

    best = max(scores, key=scores.get)
    second = sorted(scores.values(), reverse=True)
    margin = (second[0] - second[1]) if len(second) > 1 else second[0]
    confidence = float(max(0.35, min(0.95, 0.45 + 1.6 * margin)))

    # If winner is neutral but a non-neutral score is close, break ties toward prosody
    if best == "neutral":
        non_neutral = {k: v for k, v in scores.items() if k != "neutral"}
        alt, alt_s = max(non_neutral.items(), key=lambda kv: kv[1])
        if alt_s > 0.52 and alt_s + 0.04 >= scores["neutral"]:
            best = alt
            confidence = float(max(0.42, min(0.88, alt_s)))

    return best, confidence, scores


def _build_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    assert librosa is not None
    y = librosa.util.normalize(y) * 0.98
    rms, zcr, cent, rolloff = _frame_features(y, sr)

    rms = np.maximum(rms, 1e-10)
    rms_m = float(np.mean(rms))
    rms_max = float(np.max(rms)) + 1e-9
    e_mean = rms_m / rms_max
    e_std = float(np.std(rms / rms_max))

    z_m = float(np.mean(zcr))
    z_s = float(np.std(zcr))

    c_hz = float(np.mean(cent))
    c_norm = _safe_norm(c_hz, 800.0, 3800.0)

    r_hz = float(np.mean(rolloff))
    r_norm = _safe_norm(r_hz, 2000.0, 9000.0)

    f0_mean_hz, f0_std_hz, vf = _f0_stats(y, sr)
    f0_mean_norm = _safe_norm(f0_mean_hz, 95.0, 280.0) if f0_mean_hz > 0 else 0.35
    f0_shake = _safe_norm(f0_std_hz, 5.0, 45.0)

    on_m, on_s = _onset_proxy(y, sr)

    return {
        "energy_mean": float(max(0.0, min(1.0, e_mean))),
        "energy_jitter": float(max(0.0, min(1.0, e_std * 3.5))),
        "zcr_mean": float(max(0.0, min(1.0, z_m * 8.0))),
        "zcr_jitter": float(max(0.0, min(1.0, z_s * 25.0))),
        "brightness": float(max(0.0, min(1.0, c_norm))),
        "rolloff_norm": float(max(0.0, min(1.0, r_norm))),
        "f0_mean_norm": float(max(0.0, min(1.0, f0_mean_norm))),
        "f0_shake": float(max(0.0, min(1.0, f0_shake))),
        "voiced_frac": float(max(0.0, min(1.0, vf))),
        "onset_mean": float(max(0.0, min(1.0, on_m))),
        "onset_std": float(max(0.0, min(1.0, on_s * 4.0))),
    }


def detect(audio_bytes: bytes, filename: str = "audio.webm") -> dict:
    if not audio_bytes:
        return _empty()

    if librosa is None:
        return _empty()

    try:
        suffix = os.path.splitext(filename)[1] or ".webm"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            y, sr = librosa.load(tmp.name, sr=22050, mono=True)

        if y.size == 0:
            raise ValueError("empty audio")

        max_samples = int(sr * _MAX_SEC)
        if y.size > max_samples:
            y = y[:max_samples]

        feat = _build_features(y, sr)
        label, conf, scores = _score_emotions(feat)

        arousal = float(
            0.45 * feat["energy_mean"]
            + 0.25 * feat["onset_mean"]
            + 0.20 * feat["f0_shake"]
            + 0.10 * feat["brightness"]
        )
        valence = float(
            0.35 * (feat["brightness"] - 0.5) * 2.0
            + 0.35 * (feat["energy_mean"] - 0.5) * 2.0
            + 0.30 * (0.5 - feat["f0_shake"])
        )
        valence = max(-1.0, min(1.0, valence))

        stress = float(
            0.35 * feat["f0_shake"]
            + 0.30 * feat["energy_jitter"]
            + 0.20 * feat["zcr_jitter"]
            + 0.15 * feat["zcr_mean"]
        )

        intensity = "low"
        if conf >= 0.72 or stress > 0.55:
            intensity = "high"
        elif conf >= 0.52 or stress > 0.38:
            intensity = "medium"

        top_scores: List[Tuple[str, float]] = sorted(
            ((k, v) for k, v in scores.items() if k != "neutral"),
            key=lambda kv: kv[1],
            reverse=True,
        )[:3]

        return {
            "label": label,
            "intensity": intensity,
            "energy": round(feat["energy_mean"], 3),
            "pitch": round(feat["f0_mean_norm"], 3),
            "stress_score": round(stress, 3),
            "arousal": round(arousal, 3),
            "valence": round(valence, 3),
            "prosody_confidence": round(conf, 3),
            "prosody_scores": {k: round(v, 3) for k, v in top_scores},
        }
    except Exception:
        return _fallback()


def _empty() -> dict:
    return {
        "label": "neutral",
        "intensity": "low",
        "energy": 0.0,
        "pitch": 0.0,
        "stress_score": 0.0,
        "arousal": 0.0,
        "valence": 0.0,
        "prosody_confidence": 0.0,
        "prosody_scores": {},
    }


def _fallback() -> dict:
    d = _empty()
    d.update({"energy": 0.25, "pitch": 0.45, "stress_score": 0.35, "prosody_confidence": 0.2})
    return d
