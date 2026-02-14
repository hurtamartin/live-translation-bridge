import numpy as np
from scipy.signal import lfilter


def preprocess_audio(audio_np: np.ndarray, config: dict) -> np.ndarray:
    """Apply enabled preprocessing steps to audio before translation."""
    # Early return if no preprocessing is enabled — avoid unnecessary copy
    if not (config["preprocess_highpass"] or config["preprocess_noise_gate"] or config["preprocess_normalize"]):
        return audio_np

    sr = config["sample_rate"]
    audio = audio_np.copy()

    # 1. High-pass filter (remove low rumble below speech frequencies)
    if config["preprocess_highpass"]:
        cutoff = config["preprocess_highpass_cutoff"]
        rc = 1.0 / (2.0 * np.pi * cutoff)
        dt = 1.0 / sr
        alpha = rc / (rc + dt)
        b = [alpha, -alpha]
        a = [1.0, -alpha]
        audio = lfilter(b, a, audio).astype(np.float32)

    # 2. Noise gate (silence audio below threshold) — vectorized
    if config["preprocess_noise_gate"]:
        threshold_db = config["preprocess_noise_gate_threshold"]
        threshold_linear = 10.0 ** (threshold_db / 20.0)
        frame_size = int(sr * 0.02)  # 20ms frames
        n_full = len(audio) // frame_size
        if n_full > 0:
            frames = audio[:n_full * frame_size].reshape(n_full, frame_size)
            rms = np.sqrt(np.mean(frames ** 2, axis=1))
            mask = rms < threshold_linear
            frames[mask] = 0.0
            audio[:n_full * frame_size] = frames.reshape(-1)
        remainder = len(audio) - n_full * frame_size
        if remainder > 0:
            tail = audio[n_full * frame_size:]
            if np.sqrt(np.mean(tail ** 2)) < threshold_linear:
                audio[n_full * frame_size:] = 0.0

    # 3. Normalize volume
    if config["preprocess_normalize"]:
        target_db = config["preprocess_normalize_target"]
        peak = np.max(np.abs(audio))
        if peak > 1e-6:  # avoid division by zero on silence
            current_db = 20.0 * np.log10(peak)
            gain_db = target_db - current_db
            gain_linear = 10.0 ** (gain_db / 20.0)
            audio = audio * gain_linear
            # Clip to prevent distortion
            audio = np.clip(audio, -1.0, 1.0)

    return audio
