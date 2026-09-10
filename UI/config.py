"""
Centralized configuration and default state for the ANC demo.

Every value a judge sees on screen is read from this state dictionary.
When the real AI + FxLMS backend is ready, populate this same structure
from the backend's output instead of `anc_engine.anc_filter()` — nothing
in app.py needs to change.
"""

APP_TITLE = "AI-BASED ADAPTIVE NOISE CANCELLATION (ANC)"
APP_SUBTITLE = "AI/ML + FxLMS Real-Time Noise Suppression System"
APP_BADGE = "SIH"

SAMPLE_RATE = 16000
DEMO_DURATION_SEC = 4.0

AUDIO_DIR = "audio"
NOISY_PATH = f"{AUDIO_DIR}/noisy.wav"
FILTERED_PATH = f"{AUDIO_DIR}/filtered.wav"

NOISE_CATEGORIES = ["Stationary", "Non-Stationary", "Impulsive", "Mixed"]

# Status values shown in the header badge and driving card styling.
STATUS_READY = "READY"
STATUS_PROCESSING = "PROCESSING"
STATUS_ANC_ACTIVE = "ANC ACTIVE"
STATUS_FILTERED = "FILTERED"
STATUS_ERROR = "ERROR"

STATUS_COLORS = {
    STATUS_READY: "#4fd8e0",
    STATUS_PROCESSING: "#f0a83c",
    STATUS_ANC_ACTIVE: "#43cb0d",
    STATUS_FILTERED: "#43cb0d",
    STATUS_ERROR: "#ef4444",
}

COLORS = {
    "bg": "#0b1420",
    "panel": "#101c28",
    "panel_alt": "#0d1721",
    "border": "#22323e",
    "text": "#e2ecf2",
    "text_dim": "#8aa0ae",
    "cyan": "#4fd8e0",
    "green": "#43cb0d",
    "orange": "#f0a83c",
    "red": "#ef4444",
}

# This dict is the single source of truth for every metric shown on screen.
# `app.py` only ever reads/writes this structure — swap the values coming
# out of anc_engine.anc_filter() for real backend values later and the UI
# needs no changes.
DEFAULT_STATE = {
    "noise_type": "Stationary",
    "confidence": 0.0,
    "noise_level_db": 0.0,
    "noise_reduction_db": 0.0,
    "speech_intelligibility": 0.0,
    "latency_ms": 0.0,
    "cpu_usage_pct": 0.0,
    "anc_status": STATUS_READY,
    "input_audio": None,   # np.ndarray, mono float32
    "output_audio": None,  # np.ndarray, mono float32, filled after filtering
    "sample_rate": SAMPLE_RATE,
}
