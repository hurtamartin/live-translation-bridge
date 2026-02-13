import asyncio
import collections
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from src.config import runtime_config
from src.logging_handler import logger
from src.translation.session import SessionManager
from src.audio.capture import detect_device, load_vad
from src.translation.engine import load_model, warmup_model

# Audio pipeline
audio_queue = queue.Queue()
translation_queue = asyncio.Queue()
stop_event = threading.Event()
audio_stream = None
audio_stream_lock = threading.Lock()
start_time = time.time()
native_sample_rate = 16000
resampler = None

# Audio monitoring
_audio_level_db = -60.0
_audio_level_peak = -60.0
_audio_level_lock = threading.Lock()
_audio_history = collections.deque(maxlen=60)  # 60s of 1-per-second samples

# Performance metrics
_perf_metrics = {
    "encoder_ms": collections.deque(maxlen=100),
    "decoder_ms": collections.deque(maxlen=100),
    "total_translations": 0,
    "last_inference_time": 0,
}
_perf_lock = threading.Lock()

# Translation history
_translation_history = collections.deque(maxlen=50)
_translation_history_lock = threading.Lock()

# Singletons (initialized in initialize())
inference_executor = ThreadPoolExecutor(max_workers=1)
manager = SessionManager()
device = None
dtype = None
processor = None
model = None
vad_model = None
vad_utils = None


def _update_audio_history():
    """Called once per second to record audio level history."""
    with _audio_level_lock:
        _audio_history.append({"t": time.time(), "db": _audio_level_db, "peak": _audio_level_peak})


def initialize():
    """Load model, VAD, and run warmup. Called from app.py at startup."""
    global device, dtype, processor, model, vad_model, vad_utils

    device, dtype = detect_device()
    processor, model, dtype = load_model(device)
    vad_model, _vad_utils = load_vad()
    vad_utils = _vad_utils
    warmup_model(processor, model, device, dtype, runtime_config["sample_rate"])
