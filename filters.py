import numpy as np
import noisereduce as nr
from scipy.signal import butter, sosfilt

def _butter_bandpass(lowcut=300.0, highcut=3400.0, fs=16000, order=5):
    """Butterworth filter to isolate critical speech frequencies."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos

def apply_transient_limiter(audio: np.ndarray, threshold_factor: float = 2.5) -> np.ndarray:
    """
    Simulates ear protection against impulse noise (gunfire/blasts).
    Clamps spikes exceeding normal conversational levels.
    """
    # Calculate baseline RMS volume (ignoring silent frames)
    frame_size = 512
    num_frames = len(audio) // frame_size
    rms_values = [
        np.sqrt(np.mean(audio[i * frame_size : (i + 1) * frame_size] ** 2))
        for i in range(num_frames)
    ]
    avg_rms = np.median(rms_values) if rms_values else 0.05
    
    # Define dynamic ceiling based on average speaking level
    ceiling = max(avg_rms * threshold_factor, 0.15)
    
    # Soft tanh saturation clamp for smooth audio without harsh square-wave clicks
    clamped = np.tanh(audio / ceiling) * ceiling
    return clamped

def clean_tactical_audio(audio_data: np.ndarray, sample_rate: int = 16000, mode: str = "steady") -> np.ndarray:
    """
    Core Entrypoint for Group 1.
    """
    if len(audio_data) == 0:
        return audio_data

    # Ensure float32 range [-1.0, 1.0]
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)

    if mode == "steady":
        # 1. Aggressive stationary spectral subtractor for engine/generator drone
        cleaned = nr.reduce_noise(
            y=audio_data,
            sr=sample_rate,
            stationary=True,
            prop_decrease=0.90,
            n_fft=1024,
            win_length=512
        )

    elif mode == "impulse":
        # 2. Fast blast protection circuit simulation
        cleaned = apply_transient_limiter(audio_data, threshold_factor=2.2)

    elif mode == "dynamic":
        # 3. Non-stationary noise reduction with speech formant recovery
        sos = _butter_bandpass(lowcut=300, highcut=3400, fs=sample_rate)
        speech_band = sosfilt(sos, audio_data)
        
        # Attenuate background on the full spectrum and blend speech back in
        bg_reduced = nr.reduce_noise(
            y=audio_data,
            sr=sample_rate,
            stationary=False,
            prop_decrease=0.65
        )
        cleaned = 0.7 * speech_band + 0.3 * bg_reduced

    else:
        cleaned = audio_data

    # Normalization to prevent digital distortion
    peak = np.max(np.abs(cleaned))
    if peak > 0:
        cleaned = cleaned / peak

    return cleaned.astype(np.float32)