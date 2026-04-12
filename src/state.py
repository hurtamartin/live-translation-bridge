import asyncio
import collections
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from src.config import get_config_snapshot
from src.logging_handler import logger
from src.translation.session import SessionManager
from src.audio.capture import detect_device, load_vad
from src.translation.engine import load_model, warmup_model

# Audio pipeline
audio_queue = queue.Queue(maxsize=500)
translation_queue = asyncio.Queue(maxsize=int(os.environ.get("TRANSLATION_QUEUE_MAXSIZE", "100")))
stop_event = threading.Event()
translation_paused = threading.Event()  # when set, translation is paused
audio_stream = None
audio_stream_lock = threading.Lock()
audio_device_error = threading.Event()  # signaled when audio device fails
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
    "stuck_inference_count": 0,
    "audio_queue_drops": 0,
    "audio_status_drops": 0,
    "audio_buffer_overflows": 0,
    "translation_queue_drops": 0,
    "processing_loop_errors": 0,
}
_perf_lock = threading.Lock()

# Translation history
_translation_history = collections.deque(maxlen=50)
_translation_history_lock = threading.Lock()

# Singletons (initialized in initialize())
inference_executor = ThreadPoolExecutor(max_workers=1)
processing_thread = None  # set in start_server()
manager = SessionManager()
device = None
dtype = None
processor = None
model = None
vad_model = None
vad_utils = None

# Cached device name (updated in restart_audio_stream)
cached_device_name = None

# Inference pending counter (replaces _work_queue.qsize())
inference_pending = 0
inference_pending_lock = threading.Lock()
inference_stuck = threading.Event()
inference_stuck_since = 0.0
_last_stuck_log = 0.0

# Async task references (for graceful shutdown)
broadcaster_task = None
audio_ticker_task = None
watchdog_thread = None


def _update_audio_history():
    """Called once per second to record audio level history."""
    global _audio_level_peak
    with _audio_level_lock:
        _audio_history.append({"t": time.time(), "db": _audio_level_db, "peak": _audio_level_peak})
        _audio_level_peak = max(_audio_level_peak - 0.5, _audio_level_db)


def increment_metric(name: str, amount: int = 1):
    with _perf_lock:
        if name not in _perf_metrics:
            _perf_metrics[name] = 0
        _perf_metrics[name] += amount


def get_queue_stats() -> dict:
    return {
        "audio_queue_size": audio_queue.qsize(),
        "audio_queue_maxsize": audio_queue.maxsize,
        "translation_queue_size": translation_queue.qsize(),
        "translation_queue_maxsize": translation_queue.maxsize,
    }


def initialize():
    """Load model, VAD, and run warmup. Called from app.py at startup."""
    global device, dtype, processor, model, vad_model, vad_utils

    device, dtype = detect_device()
    processor, model, dtype = load_model(device)
    vad_model, _vad_utils = load_vad()
    vad_utils = _vad_utils
    warmup_model(processor, model, device, dtype, get_config_snapshot()["sample_rate"])
