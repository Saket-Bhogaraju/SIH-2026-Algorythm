"""
ANC processing logic — kept completely separate from app.py.

`anc_filter()` is the one function the UI calls. Today it runs an in-browser
adaptive spectral-gating filter (via `noisereduce`) as a stand-in for the
real AI-driven FxLMS system. To integrate the real backend later, replace
the body of `anc_filter()` with a call to your trained model / FxLMS
pipeline and keep the same return dictionary shape — nothing in app.py
needs to change.

    filtered_audio = anc_filter(noisy_audio, sample_rate)
"""

from __future__ import annotations

import time

import numpy as np
import noisereduce as nr
import psutil

from audio_utils import normalize, rms_db


class FilterError(Exception):
    """Raised when noise-cancellation processing fails."""


def detect_noise_type(y: np.ndarray, sr: int) -> tuple[str, float]:
    """Lightweight heuristic classifier based on crest factor (impulsiveness)
    and short-term energy variance (how non-stationary the noise is).

    Returns (label, confidence_percent). This is a stand-in for the trained
    AI noise classifier — swap it out once that model is ready, keeping the
    same return type.
    """
    frame, hop = 2048, 1024
    frames = [y[i:i + frame] for i in range(0, max(len(y) - frame, 0), hop)]
    if not frames:
        frames = [y]

    rms_vals = np.array([np.sqrt(np.mean(np.square(f))) + 1e-9 for f in frames])
    peak_vals = np.array([np.max(np.abs(f)) if len(f) else 0.0 for f in frames])
    crest = float(np.mean(peak_vals / rms_vals))
    variance = float(np.var(rms_vals) / (np.mean(rms_vals) ** 2 + 1e-9))

    impulsive = float(np.clip((crest - 4.0) / 6.0, 0, 1))
    non_stationary = float(np.clip(variance * 8.0, 0, 1) * (1 - impulsive * 0.5))
    stationary = float(np.clip(1 - impulsive - non_stationary, 0.05, 1))

    scores = {"Impulsive": impulsive, "Non-Stationary": non_stationary, "Stationary": stationary}
    ranked = sorted(scores.values(), reverse=True)
    total = sum(scores.values()) + 1e-9

    if len(ranked) >= 2 and (ranked[0] - ranked[1]) < 0.12 and ranked[1] > 0.25:
        label = "Mixed"
        confidence = (ranked[0] + ranked[1]) / 2 / total
    else:
        label = max(scores, key=scores.get)
        confidence = scores[label] / total

    return label, round(float(confidence) * 100, 1)


def anc_filter(y: np.ndarray, sr: int, suppression: float = 0.75, ai_detection: bool = True) -> dict:
    """Run adaptive noise cancellation on a mono audio clip.

    Args:
        y: mono float32 audio samples, noisy input.
        sr: sample rate in Hz.
        suppression: 0.0-1.0, how aggressively to remove noise.
        ai_detection: whether to run the noise-type classifier.

    Returns:
        dict matching config.DEFAULT_STATE's metric keys, plus "output_audio".
    """
    if y is None or len(y) == 0:
        raise FilterError("No input audio was provided.")
    if not np.isfinite(y).all():
        raise FilterError("Input audio contains invalid samples.")

    prop_decrease = float(np.clip(suppression, 0.05, 1.0))

    t0 = time.perf_counter()
    try:
        filtered = nr.reduce_noise(y=y, sr=sr, stationary=False, prop_decrease=prop_decrease)
    except Exception as exc:  # noqa: BLE001
        raise FilterError(f"Filter processing failed: {exc}") from exc
    latency_ms = (time.perf_counter() - t0) * 1000

    cpu_usage = psutil.cpu_percent(interval=0.05)

    noise_before = rms_db(y)
    noise_after = rms_db(filtered)
    noise_reduction_db = round(max(noise_before - noise_after, 0.0), 1)

    # Speech-intelligibility proxy: waveform correlation between input and
    # output, scaled to a plausible percentage. This is a demo stand-in —
    # swap for a real STOI/PESQ score against a clean reference once the
    # trained model + evaluation pipeline is wired in.
    n = min(len(y), len(filtered))
    if n > 1 and np.std(y[:n]) > 1e-6 and np.std(filtered[:n]) > 1e-6:
        corr = float(np.corrcoef(y[:n], filtered[:n])[0, 1])
    else:
        corr = 0.0
    speech_intelligibility = round(float(np.clip(60 + corr * 35 + prop_decrease * 5, 0, 100)), 1)

    noise_type, confidence = detect_noise_type(y, sr) if ai_detection else ("N/A", 0.0)

    return {
        "output_audio": normalize(filtered),
        "noise_type": noise_type,
        "confidence": confidence,
        "noise_level_db": round(noise_before, 1),
        "noise_reduction_db": noise_reduction_db,
        "speech_intelligibility": speech_intelligibility,
        "latency_ms": round(latency_ms, 1),
        "cpu_usage_pct": round(cpu_usage, 1),
    }
