import queue

import numpy as np
import sounddevice as sd
import torch
import torchaudio

from src.logging_handler import logger
from src.config import runtime_config


def detect_device():
    """Auto-detect best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.float16
        logger.info(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
        return device, dtype
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float16
        logger.info("Using device: MPS (Apple Silicon)")
        return device, dtype
    logger.warning("Using CPU - translation will be slow")
    device = torch.device("cpu")
    dtype = torch.float32
    return device, dtype


def load_vad():
    """Load Silero VAD model. Returns (vad_model, vad_utils)."""
    logger.info("Loading Silero VAD...")
    vad_model, vad_utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        trust_repo=True
    )
    logger.info("Silero VAD loaded.")
    return vad_model, vad_utils


def is_speech(audio_chunk_np: np.ndarray, vad_model, sample_rate: int) -> bool:
    """Check if audio chunk contains speech using Silero VAD."""
    tensor = torch.from_numpy(audio_chunk_np).float()
    if tensor.dim() > 1:
        tensor = tensor.squeeze()
    confidence = vad_model(tensor, sample_rate).item()
    return confidence > 0.5


def compute_audio_level(audio_np: np.ndarray) -> float:
    """Compute RMS audio level in dB, clamped to -60 dB."""
    rms = np.sqrt(np.mean(audio_np ** 2))
    if rms < 1e-10:
        return -60.0
    db = 20.0 * np.log10(rms)
    return float(max(db, -60.0))


def create_resampler(orig_freq: int, new_freq: int = 16000):
    if orig_freq == new_freq:
        return None
    return torchaudio.transforms.Resample(orig_freq=orig_freq, new_freq=new_freq)


def resample_audio(audio_np: np.ndarray, resampler_obj) -> np.ndarray:
    if resampler_obj is None:
        return audio_np
    tensor = torch.from_numpy(audio_np).float().unsqueeze(0)
    resampled = resampler_obj(tensor)
    return resampled.squeeze(0).numpy()


def get_audio_devices() -> list[dict]:
    """Get list of available audio input devices."""
    devices = sd.query_devices()
    result = []
    default_input = sd.default.device[0] if isinstance(sd.default.device, tuple) else sd.default.device
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            result.append({
                "index": i,
                "name": dev['name'],
                "max_input_channels": dev['max_input_channels'],
                "default_samplerate": dev['default_samplerate'],
                "is_default": (i == default_input),
            })
    return result


def restart_audio_stream(device_index, channel, state):
    """Restart audio stream with new device/channel settings.

    state must have: audio_stream, audio_stream_lock, native_sample_rate,
                     resampler, audio_queue, runtime_config
    """
    with state.audio_stream_lock:
        if state.audio_stream is not None:
            try:
                state.audio_stream.stop()
            except Exception as e:
                logger.warning(f"Error stopping audio stream: {e}")
            try:
                state.audio_stream.close()
            except Exception as e:
                logger.warning(f"Error closing audio stream: {e}")
            logger.info("Previous audio stream stopped.")
            state.audio_stream = None

        # Determine native sample rate of the device
        if device_index is not None:
            dev_info = sd.query_devices(device_index)
            native_sr = int(dev_info['default_samplerate'])
            max_ch = dev_info['max_input_channels']
            channels_to_open = max_ch
        else:
            dev_info = sd.query_devices(kind='input')
            native_sr = int(dev_info['default_samplerate'])
            channels_to_open = 1

        state.native_sample_rate = native_sr

        # Create resampler if needed (native -> 16kHz)
        target_sr = runtime_config["sample_rate"]
        state.resampler = create_resampler(native_sr, target_sr)
        if state.resampler is not None:
            logger.info(f"Resampling active: {native_sr} Hz -> {target_sr} Hz")
        else:
            logger.info(f"No resampling needed: device already at {native_sr} Hz")

        # If channel > 0, we need to open enough channels
        if channel > 0:
            channels_to_open = max(channels_to_open, channel + 1)

        audio_queue = state.audio_queue
        device_error = state.audio_device_error

        def audio_callback(indata, frames, time_info, status):
            try:
                if status:
                    logger.warning(f"Audio status: {status}")
                    if status.input_overflow:
                        return  # drop chunk on overflow, don't signal error
                    if status.input_underflow or "error" in str(status).lower():
                        device_error.set()
                        return
                audio_queue.put_nowait(indata.copy())
            except queue.Full:
                pass  # drop chunk if queue full — better than blocking callback
            except Exception as e:
                logger.error(f"Audio callback error: {e}")
                device_error.set()

        def multi_channel_callback(indata, frames, time_info, status):
            try:
                if status:
                    logger.warning(f"Audio status: {status}")
                    if status.input_overflow:
                        return
                    if status.input_underflow or "error" in str(status).lower():
                        device_error.set()
                        return
                ch = runtime_config["audio_channel"]
                if ch < indata.shape[1]:
                    audio_queue.put_nowait(indata[:, ch:ch+1].copy())
                else:
                    audio_queue.put_nowait(indata[:, 0:1].copy())
            except queue.Full:
                pass
            except Exception as e:
                logger.error(f"Audio callback error: {e}")
                device_error.set()

        try:
            if channel > 0 or (device_index is not None and channels_to_open > 1):
                state.audio_stream = sd.InputStream(
                    device=device_index,
                    channels=channels_to_open,
                    samplerate=native_sr,
                    callback=multi_channel_callback,
                )
            else:
                state.audio_stream = sd.InputStream(
                    device=device_index,
                    channels=1,
                    samplerate=native_sr,
                    callback=audio_callback,
                )
            state.audio_stream.start()
            dev_name = "default" if device_index is None else sd.query_devices(device_index)['name']
            state.cached_device_name = dev_name
            logger.info(f"Audio stream started: device={dev_name}, channel={channel}, native_sr={native_sr}")
        except Exception as e:
            logger.error(f"Failed to start audio stream: {e}")
            state.audio_stream = None
            raise
