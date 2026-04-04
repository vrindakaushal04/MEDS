"""Audio-side emotion cues from simple prosody features (Librosa) or safe defaults."""

import io
import os
import tempfile


def _map_features(rms: float, zcr: float, centroid_norm: float) -> dict:
    energy = min(1.0, float(rms) * 25.0)
    stress = min(1.0, float(zcr) * 2.0 + abs(centroid_norm - 0.5) * 0.5)
    pitch_proxy = float(centroid_norm)

    if energy > 0.65 and stress > 0.55:
        label, intensity = "agitated", "high"
    elif stress > 0.6 and energy < 0.35:
        label, intensity = "distressed", "high"
    elif energy < 0.25:
        label, intensity = "low_mood", "medium"
    elif energy > 0.5:
        label, intensity = "happy", "medium"
    else:
        label, intensity = "neutral", "low"

    return {
        "label": label,
        "intensity": intensity,
        "energy": round(energy, 3),
        "pitch": round(pitch_proxy, 3),
        "stress_score": round(stress, 3),
    }


def detect(audio_bytes: bytes, filename: str = "audio.webm") -> dict:
    if not audio_bytes:
        return {
            "label": "neutral",
            "intensity": "low",
            "energy": 0.0,
            "pitch": 0.0,
            "stress_score": 0.0,
        }

    try:
        import librosa
        import numpy as np

        suffix = os.path.splitext(filename)[1] or ".webm"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            y, sr = librosa.load(tmp.name, sr=16000, mono=True)
        if y.size == 0:
            raise ValueError("empty audio")

        rms = float(np.sqrt(np.mean(np.square(y))))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        centroid_norm = float(np.mean(cent) / (sr / 2)) if sr else 0.5
        centroid_norm = max(0.0, min(1.0, centroid_norm))

        return _map_features(rms, zcr, centroid_norm)
    except Exception:
        return {
            "label": "neutral",
            "intensity": "low",
            "energy": 0.35,
            "pitch": 0.5,
            "stress_score": 0.25,
        }
