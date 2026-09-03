import numpy as np
from filters import clean_tactical_audio

def calculate_snr(clean_signal: np.ndarray, noisy_signal: np.ndarray) -> float:
    """
    Computes Signal-to-Noise Ratio (SNR) in decibels (dB).
    Formula: 10 * log10(Power_signal / Power_noise)
    """
    # Noise = (noisy or processed signal) - reference clean signal
    noise = noisy_signal - clean_signal
    
    p_signal = np.mean(clean_signal ** 2)
    p_noise = np.mean(noise ** 2)
    
    if p_noise == 0:
        return float("inf")
    if p_signal == 0:
        return float("-inf")
        
    snr_db = 10 * np.log10(p_signal / (p_noise + 1e-10))
    return float(snr_db)

def calculate_attenuation_db(original_noisy: np.ndarray, cleaned: np.ndarray) -> float:
    """
    Measures total RMS energy reduction across the entire waveform.
    Formula: 20 * log10(RMS_before / RMS_after)
    """
    rms_before = np.sqrt(np.mean(original_noisy ** 2))
    rms_after = np.sqrt(np.mean(cleaned ** 2))
    
    if rms_after == 0:
        return float("inf")
    
    attenuation_db = 20 * np.log10((rms_before + 1e-10) / (rms_after + 1e-10))
    return float(attenuation_db)

def run_benchmark():
    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # 1. Generate Synthetic Voice Reference (300Hz, 700Hz, 1500Hz)
    clean_voice = 0.3 * np.sin(2 * np.pi * 300 * t) + \
                  0.2 * np.sin(2 * np.pi * 700 * t) + \
                  0.1 * np.sin(2 * np.pi * 1500 * t)
    
    # 2. Benchmark Cases
    scenarios = {
        "steady": {
            "name": "Engine / Generator Drone (120 Hz Hum)",
            "noise": 0.5 * np.sin(2 * np.pi * 120 * t) + 0.05 * np.random.normal(0, 1, len(t))
        },
        "dynamic": {
            "name": "Tactical Siren / Modulated Hum",
            "noise": 0.4 * np.sin(2 * np.pi * (800 + 200 * np.sin(2 * np.pi * 2 * t)) * t)
        },
        "impulse": {
            "name": "High-Energy Gunfire / Blast Transient",
            "noise": np.zeros_like(t)
        }
    }
    # Add blast spikes for impulse test
    scenarios["impulse"]["noise"][int(1.0 * sr) : int(1.05 * sr)] = 2.0 * np.random.normal(0, 1, int(0.05 * sr))
    scenarios["impulse"]["noise"][int(2.2 * sr) : int(2.25 * sr)] = 2.0 * np.random.normal(0, 1, int(0.05 * sr))

    print("\n" + "=" * 65)
    print(" 🎖️ SIH26052 TACTICAL ACOUSTIC FILTER: BENCHMARK SUITE")
    print("=" * 65)

    for mode, data in scenarios.items():
        raw_noisy = clean_voice + data["noise"]
        
        # Run filter
        cleaned = clean_tactical_audio(raw_noisy, sample_rate=sr, mode=mode)
        
        # Metrics
        initial_snr = calculate_snr(clean_voice, raw_noisy)
        final_snr = calculate_snr(clean_voice, cleaned)
        snr_gain = final_snr - initial_snr
        attenuation = calculate_attenuation_db(raw_noisy, cleaned)
        
        print(f"\nScenario: [{mode.upper()}] - {data['name']}")
        print(f" • Input SNR        : {initial_snr:+.2f} dB")
        print(f" • Output SNR       : {final_snr:+.2f} dB")
        print(f" • SNR Improvement  : {snr_gain:+.2f} dB")
        print(f" • Noise Attenuation: {attenuation:.2f} dB reduction")

    print("\n" + "=" * 65)
    print(" Benchmark completed successfully.")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    run_benchmark()