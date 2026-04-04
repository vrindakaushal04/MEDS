"""Prosody-based emotions from voice: works even when pitch tracking (F0) is weak (WebM / mic)."""

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


def _load_audio(path: str) -> Tuple[np.ndarray, int]:
    assert librosa is not None
    y, sr = librosa.load(path, sr=22050, mono=True)
    return y, int(sr)


def _frame_features(y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rms = librosa.feature.rms(y=y, frame_length=_FRAME, hop_length=_HOP)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=_FRAME, hop_length=_HOP)[0]
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=_HOP)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=_HOP)[0]
    return rms, zcr, cent, rolloff


def _f0_stats(y: np.ndarray, sr: int) -> Tuple[float, float, float]:
    fmin = librosa.note_to_hz("C2")
    fmax = librosa.note_to_hz("C7")
    f0, vf, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, hop_length=_HOP)
    voiced = vf.astype(bool) & np.isfinite(f0)
    if not np.any(voiced):
        return 0.0, 0.0, 0.0
    fv = f0[voiced]
    return float(np.nanmean(fv)), float(np.nanstd(fv)), float(np.mean(voiced))


def _f0_stats_safe(y: np.ndarray, sr: int) -> Tuple[float, float, float]:
    try:
        return _f0_stats(y, sr)
    except Exception:
        return 0.0, 0.0, 0.0


def _onset_proxy(y: np.ndarray, sr: int) -> Tuple[float, float]:
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=_HOP)
    if env.size == 0:
        return 0.0, 0.0
    e = env / (float(np.max(env)) + 1e-9)
    return float(np.mean(e)), float(np.std(e))


def _spectral_shake_proxy(rms: np.ndarray, cent: np.ndarray) -> float:
    """When F0 is missing, use loudness + timbre modulation as 'instability' proxy."""
    r = rms / (float(np.max(rms)) + 1e-9)
    r_std = float(np.std(r))
    if cent.size > 2:
        c = cent / (float(np.mean(cent)) + 1e-9)
        d = np.diff(c)
        c_std = float(np.std(np.abs(d)))
    else:
        c_std = 0.0
    raw = r_std * 1.8 + c_std * 0.9
    return float(max(0.0, min(1.0, raw * 3.2)))


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

    f0_mean_hz, f0_std_hz, vf = _f0_stats_safe(y, sr)
    shake_proxy = _spectral_shake_proxy(rms, cent)

    f0_mean_norm = _safe_norm(f0_mean_hz, 95.0, 280.0) if f0_mean_hz > 0 else float(0.35 + 0.45 * c_norm)
    f0_shake = _safe_norm(f0_std_hz, 5.0, 45.0) if f0_mean_hz > 0 else 0.0
    f0_shake = float(max(f0_shake, shake_proxy * (0.88 if vf < 0.22 else 0.55)))

    if vf < 0.25:
        f0_shake = float(max(f0_shake, shake_proxy * 0.95))

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
        "voiced_frac": float(max(0.0, min(1.0, max(vf, shake_proxy * 0.35)))),
        "onset_mean": float(max(0.0, min(1.0, on_m))),
        "onset_std": float(max(0.0, min(1.0, on_s * 4.0))),
        "shake_proxy": shake_proxy,
    }


def _score_emotions(feat: Dict[str, float], rms_full: float) -> Tuple[str, float, Dict[str, float]]:
    e = feat["energy_mean"]
    e_std = feat["energy_jitter"]
    z = feat["zcr_mean"]
    z_std = feat["zcr_jitter"]
    c = feat["brightness"]
    f0m = feat["f0_mean_norm"]
    f0s = feat["f0_shake"]
    vf = feat["voiced_frac"]
    on_m = feat["onset_mean"]
    on_s = feat["onset_std"]

    # True silence / near-empty clip
    if rms_full < 0.003 and e < 0.04:
        return "neutral", 0.3, {"neutral": 1.0}

    scores: Dict[str, float] = {}

    # Lowered midpoints so real mic speech (often quieter/shorter) triggers emotions
    scores["agitated"] = (
        0.32 * _sigmoid(e, k=10, mid=0.26)
        + 0.22 * _sigmoid(z, k=12, mid=0.08)
        + 0.18 * _sigmoid(c, k=10, mid=0.35)
        + 0.18 * _sigmoid(on_m, k=10, mid=0.22)
        + 0.10 * _sigmoid(e_std, k=12, mid=0.14)
    )

    scores["anxious"] = (
        0.34 * _sigmoid(f0s, k=11, mid=0.10)
        + 0.28 * _sigmoid(e_std, k=11, mid=0.12)
        + 0.18 * _sigmoid(z_std, k=11, mid=0.05)
        + 0.12 * _sigmoid(on_s, k=10, mid=0.08)
        + 0.08 * _sigmoid(1.0 - e, k=8, mid=0.45)
    )

    scores["sad"] = (
        0.30 * _sigmoid(1.0 - e, k=10, mid=0.38)
        + 0.28 * _sigmoid(1.0 - f0s, k=8, mid=0.42)
        + 0.22 * _sigmoid(1.0 - c, k=10, mid=0.38)
        + 0.12 * _sigmoid(1.0 - f0m, k=8, mid=0.38)
        + 0.08 * _sigmoid(1.0 - on_m, k=8, mid=0.45)
    )

    scores["low_mood"] = (
        0.38 * _sigmoid(1.0 - e, k=9, mid=0.32)
        + 0.28 * _sigmoid(1.0 - on_m, k=9, mid=0.38)
        + 0.18 * _sigmoid(1.0 - c, k=8, mid=0.38)
        + 0.16 * _sigmoid(0.22 - abs(f0s - 0.20), k=14, mid=0.0)
    )

    scores["happy"] = (
        0.28 * _sigmoid(e, k=9, mid=0.26)
        + 0.26 * _sigmoid(f0s, k=9, mid=0.14)
        + 0.22 * _sigmoid(c, k=8, mid=0.30)
        + 0.14 * _sigmoid(on_m, k=9, mid=0.22)
        + 0.10 * (1.0 - _sigmoid(e_std, k=12, mid=0.32))
    )

    scores["distressed"] = (
        0.30 * _sigmoid(e_std, k=10, mid=0.15)
        + 0.26 * _sigmoid(f0s, k=10, mid=0.15)
        + 0.22 * _sigmoid(1.0 - e, k=8, mid=0.38)
        + 0.22 * _sigmoid(z_std, k=10, mid=0.06)
    )

    scores["angry"] = (
        0.30 * _sigmoid(e, k=11, mid=0.30)
        + 0.24 * _sigmoid(z, k=12, mid=0.09)
        + 0.22 * _sigmoid(c, k=10, mid=0.38)
        + 0.14 * _sigmoid(on_m, k=10, mid=0.26)
        + 0.10 * _sigmoid(e_std, k=10, mid=0.18)
    )

    max_nn = max(v for k, v in scores.items())
    # Make neutral harder to win — reduce its weight significantly
    scores["neutral"] = (
        0.18 * (1.0 - max_nn)
        + 0.08 * (1.0 - vf)
        + 0.07 * (1.0 - on_m)
        + 0.05 * (1.0 - e_std)
    )

    best = max(scores, key=scores.get)
    ordered = sorted(scores.values(), reverse=True)
    margin = (ordered[0] - ordered[1]) if len(ordered) > 1 else ordered[0]
    confidence = float(max(0.40, min(0.95, 0.45 + 1.8 * margin)))

    # Anti-neutral override: if any emotion is reasonably strong, prefer it
    if best == "neutral":
        non_neutral = {k: v for k, v in scores.items() if k != "neutral"}
        alt, alt_s = max(non_neutral.items(), key=lambda kv: kv[1])
        # Lowered threshold from 0.36 to 0.22 — real speech should trigger this
        if alt_s >= 0.22:
            best = alt
            confidence = float(max(0.44, min(0.92, 0.42 + alt_s)))

    return best, confidence, scores


def _refine_label(label: str, scores: Dict[str, float], stress: float) -> str:
    if label != "neutral" or stress < 0.4:
        return label
    nn = {k: v for k, v in scores.items() if k != "neutral"}
    if not nn:
        return label
    alt, alt_s = max(nn.items(), key=lambda kv: kv[1])
    if alt_s >= 0.4:
        return alt
    return label


def detect(audio_bytes: bytes, filename: str = "audio.webm") -> dict:
    if not audio_bytes or librosa is None:
        return _empty()

    suffix = os.path.splitext(filename)[1] or ".webm"
    tmp_path = None
    try:
        # Windows fix: delete=False so librosa can open the file after we close it
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        y, sr = _load_audio(tmp_path)

        if y.size == 0:
            return _empty()

        max_samples = int(sr * _MAX_SEC)
        if y.size > max_samples:
            y = y[:max_samples]

        rms_full = float(np.sqrt(np.mean(np.square(y))))

        feat = _build_features(y, sr)
        label, conf, scores = _score_emotions(feat, rms_full)

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

        label = _refine_label(label, scores, stress)

        intensity = "low"
        if conf >= 0.72 or stress > 0.55:
            intensity = "high"
        elif conf >= 0.52 or stress > 0.38:
            intensity = "medium"

        top_scores: List[Tuple[str, float]] = sorted(
            ((k, v) for k, v in scores.items() if k != "neutral"),
            key=lambda kv: kv[1],
            reverse=True,
        )[:4]

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
    except Exception as exc:
        import traceback
        traceback.print_exc()  # Print to console so we can see what's failing
        return _empty()
    finally:
        # Clean up temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


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
