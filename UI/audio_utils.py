"""
Audio I/O helpers. Nothing in this file knows about Streamlit or about the
ANC/AI algorithm — it only loads, generates, converts, and measures audio.
"""

from __future__ import annotations

import base64
import io
import os

import numpy as np
import soundfile as sf


class AudioError(Exception):
    """Raised for any problem loading or validating an audio clip."""


def ensure_mono(y: np.ndarray) -> np.ndarray:
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y.astype(np.float32)


def normalize(y: np.ndarray, peak: float = 0.9) -> np.ndarray:
    m = float(np.max(np.abs(y))) if len(y) else 0.0
    if m < 1e-9:
        return y.astype(np.float32)
    return (y / m * peak).astype(np.float32)


def rms_db(y: np.ndarray) -> float:
    """A relative, un-calibrated dB reading (not true SPL) — good enough to
    show a realistic-looking noise-level number in a demo. Once the system
    is wired to a calibrated microphone, replace this with real SPL."""
    rms = float(np.sqrt(np.mean(np.square(y)))) + 1e-9
    db = 20 * np.log10(rms) + 100  # shift into a familiar-looking 0-100+ range
    return float(np.clip(db, 0, 120))


def load_audio_file(file_like) -> tuple[np.ndarray, int]:
    """Load an uploaded/user-supplied audio file. Raises AudioError with a
    judge-friendly message on any failure instead of crashing the app."""
    try:
        y, sr = sf.read(file_like, dtype="float32")
    except Exception as exc:  # noqa: BLE001 - we want to catch every decode failure
        raise AudioError(
            "Unsupported or corrupted audio file. Please upload a WAV, MP3, FLAC, or OGG file."
        ) from exc

    y = ensure_mono(y)
    if y.size == 0 or not np.isfinite(y).all():
        raise AudioError("That file did not contain valid audio data.")
    return y, sr


def load_or_create_demo(path: str, sr: int, duration: float) -> tuple[np.ndarray, int]:
    """Load the bundled demo clip from disk, generating it once if missing.
    This is what fills audio/noisy.wav on first run, per the project's
    audio/ folder convention."""
    if os.path.exists(path):
        try:
            return load_audio_file(path)
        except AudioError:
            pass  # fall through and regenerate a fresh demo clip
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    y = generate_demo_noisy(sr=sr, duration=duration)
    sf.write(path, y, sr)
    return y, sr


def generate_demo_noisy(sr: int = 16000, duration: float = 4.0) -> np.ndarray:
    """A synthetic 'dirty' clip: speech-like AM tones plus steady hiss and a
    few sharp impulsive bursts — stands in for a real noisy defence-comms
    recording so the demo works with zero setup."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    speech = (
        0.35 * np.sin(2 * np.pi * 220 * t) * np.clip(np.sin(2 * np.pi * 2 * t), 0, 1)
        + 0.25 * np.sin(2 * np.pi * 440 * t) * np.clip(np.sin(2 * np.pi * 1.3 * t + 1), 0, 1)
    )
    rng = np.random.default_rng(7)
    noise = 0.25 * rng.standard_normal(len(t))
    for pos in (0.8, 1.9, 3.1):
        idx = int(pos * sr)
        span = int(0.01 * sr)
        noise[idx: idx + span] += rng.standard_normal(span) * 1.5
    dirty = speech + noise
    return normalize(dirty)


def to_wav_bytes(y: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    return buf.getvalue()


def autoplay_html(y: np.ndarray, sr: int) -> str:
    """Return an <audio autoplay> snippet so a button press can trigger
    real playback immediately, in addition to the normal player widgets."""
    b64 = base64.b64encode(to_wav_bytes(y, sr)).decode()
    return f'<audio autoplay="true" style="display:none"><source src="data:audio/wav;base64,{b64}" type="audio/wav"></audio>'
