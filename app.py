import asyncio
import threading
import queue
import time
import json
import uuid
import logging
import collections
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import sounddevice as sd
import torch
import torchaudio
from transformers import SeamlessM4Tv2ForSpeechToText, AutoProcessor
from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from scipy.signal import lfilter
import uvicorn

# --- LOGGING SETUP ---
logger = logging.getLogger("app")
logger.setLevel(logging.DEBUG)

class LogBufferHandler(logging.Handler):
    """Custom handler that stores log records in a deque for WebSocket streaming."""
    def __init__(self, maxlen=500):
        super().__init__()
        self.buffer = collections.deque(maxlen=maxlen)
        self.listeners: list[asyncio.Queue] = []
        self._lock = threading.Lock()

    def emit(self, record):
        entry = {
            "time": time.strftime("%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "message": self.format(record),
        }
        with self._lock:
            self.buffer.append(entry)
            for q in self.listeners:
                try:
                    q.put_nowait(entry)
                except asyncio.QueueFull:
                    pass

    def add_listener(self) -> asyncio.Queue:
        q = asyncio.Queue(maxsize=100)
        with self._lock:
            self.listeners.append(q)
        return q

    def remove_listener(self, q: asyncio.Queue):
        with self._lock:
            try:
                self.listeners.remove(q)
            except ValueError:
                pass

    def get_history(self) -> list[dict]:
        with self._lock:
            return list(self.buffer)

log_handler = LogBufferHandler()
log_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(log_handler)

# Also log to console
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(console_handler)

# --- RUNTIME CONFIG ---
DEFAULT_CONFIG = {
    "audio_device_index": None,
    "audio_channel": 0,
    "sample_rate": 16000,
    "silence_duration": 0.8,
    "min_chunk_duration": 1.5,
    "max_chunk_duration": 12.0,
    "context_overlap": 0.5,
    "default_target_lang": "ces",
    # Audio preprocessing
    "preprocess_noise_gate": False,
    "preprocess_noise_gate_threshold": -40.0,  # dB
    "preprocess_normalize": False,
    "preprocess_normalize_target": -3.0,  # dB
    "preprocess_highpass": False,
    "preprocess_highpass_cutoff": 80,  # Hz
    "preprocess_auto_language": False,
}

runtime_config = dict(DEFAULT_CONFIG)

# Validation ranges for config values
CONFIG_RANGES = {
    "silence_duration": (0.3, 3.0),
    "min_chunk_duration": (0.5, 5.0),
    "max_chunk_duration": (5.0, 30.0),
    "context_overlap": (0.0, 2.0),
    "sample_rate": (8000, 48000),
    "audio_channel": (0, 127),
    "preprocess_noise_gate_threshold": (-60.0, -10.0),
    "preprocess_normalize_target": (-20.0, 0.0),
    "preprocess_highpass_cutoff": (20, 300),
}

# --- GLOBAL STATE ---
audio_queue = queue.Queue()
translation_queue = asyncio.Queue()
stop_event = threading.Event()
audio_stream = None
audio_stream_lock = threading.Lock()
start_time = time.time()
native_sample_rate = 16000
resampler = None
_audio_level_db = -60.0
_audio_level_lock = threading.Lock()

def compute_audio_level(audio_np: np.ndarray) -> float:
    """Compute RMS audio level in dB, clamped to -60 dB."""
    rms = np.sqrt(np.mean(audio_np ** 2))
    if rms < 1e-10:
        return -60.0
    db = 20.0 * np.log10(rms)
    return float(max(db, -60.0))

# --- DEVICE DETECTION ---
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

device, dtype = detect_device()

# --- MODEL (direct, not pipeline) ---
MODEL_NAME = "facebook/seamless-m4t-v2-large"
logger.info(f"Loading model {MODEL_NAME}...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = SeamlessM4Tv2ForSpeechToText.from_pretrained(MODEL_NAME)
model = model.to(device=device, dtype=dtype)
model.eval()

# torch.compile on CUDA if available
if device.type == "cuda":
    try:
        model = torch.compile(model)
        logger.info("Model compiled with torch.compile()")
    except Exception as e:
        logger.warning(f"torch.compile() not available: {e}")

logger.info("Model loaded.")

# --- MODEL WARM-UP ---
def warmup_model():
    """Run dummy inference to eliminate cold-start delay for main languages."""
    logger.info("Warming up model (ces, spa, eng)...")
    sr = runtime_config["sample_rate"]
    dummy_audio = np.zeros(sr * 2, dtype=np.float32)
    inputs = processor(audio=dummy_audio, sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to(device=device, dtype=dtype) if v.dtype.is_floating_point else v.to(device=device)
              for k, v in inputs.items()}
    encoder_kwargs = {k: v for k, v in inputs.items() if k in ('input_features', 'attention_mask')}
    with torch.no_grad():
        encoder_out = model.get_encoder()(**encoder_kwargs)
        for lang in ("ces", "spa", "eng"):
            model.generate(**inputs, encoder_outputs=encoder_out, tgt_lang=lang)
    logger.info("Warm-up complete (ces, spa, eng).")

warmup_model()

# --- SILERO VAD ---
logger.info("Loading Silero VAD...")
vad_model, vad_utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    trust_repo=True
)
(get_speech_timestamps, _, read_audio, _, _) = vad_utils
logger.info("Silero VAD loaded.")

def is_speech(audio_chunk_np: np.ndarray) -> bool:
    """Check if audio chunk contains speech using Silero VAD."""
    tensor = torch.from_numpy(audio_chunk_np).float()
    if tensor.dim() > 1:
        tensor = tensor.squeeze()
    confidence = vad_model(tensor, runtime_config["sample_rate"]).item()
    return confidence > 0.5

# --- AUDIO RESAMPLING ---
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

# --- AUDIO PREPROCESSING ---
def preprocess_audio(audio_np: np.ndarray) -> np.ndarray:
    """Apply enabled preprocessing steps to audio before translation."""
    sr = runtime_config["sample_rate"]
    audio = audio_np.copy()

    # 1. High-pass filter (remove low rumble below speech frequencies)
    if runtime_config["preprocess_highpass"]:
        cutoff = runtime_config["preprocess_highpass_cutoff"]
        rc = 1.0 / (2.0 * np.pi * cutoff)
        dt = 1.0 / sr
        alpha = rc / (rc + dt)
        b = [alpha, -alpha]
        a = [1.0, -alpha]
        audio = lfilter(b, a, audio).astype(np.float32)

    # 2. Noise gate (silence audio below threshold) — vectorized
    if runtime_config["preprocess_noise_gate"]:
        threshold_db = runtime_config["preprocess_noise_gate_threshold"]
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
    if runtime_config["preprocess_normalize"]:
        target_db = runtime_config["preprocess_normalize_target"]
        peak = np.max(np.abs(audio))
        if peak > 1e-6:  # avoid division by zero on silence
            current_db = 20.0 * np.log10(peak)
            gain_db = target_db - current_db
            gain_linear = 10.0 ** (gain_db / 20.0)
            audio = audio * gain_linear
            # Clip to prevent distortion
            audio = np.clip(audio, -1.0, 1.0)

    return audio

# --- MULTI-LANGUAGE TRANSLATION ---
inference_executor = ThreadPoolExecutor(max_workers=1)

def detect_source_language(audio_np: np.ndarray) -> str | None:
    """Detect source language using the model. Returns language code or None."""
    try:
        sr = runtime_config["sample_rate"]
        inputs = processor(audio=audio_np, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(device=device, dtype=dtype) if v.dtype.is_floating_point else v.to(device=device)
                  for k, v in inputs.items()}
        with torch.no_grad():
            # Generate with English target just to get encoder output, then check predicted language
            output = model.generate(**inputs, tgt_lang="eng",
                                     return_dict_in_generate=True, output_scores=True)
        # The model's src_lang is set during processing
        src_lang = getattr(processor, '_src_lang', None)
        if src_lang:
            logger.info(f"Detected source language: {src_lang}")
        return src_lang
    except Exception as e:
        logger.warning(f"Language detection failed: {e}")
        return None

def translate_audio(audio_np: np.ndarray, target_langs: list[str]) -> dict[str, str]:
    """Translate audio to multiple languages. Encoder runs once, decoder per language."""
    processed_audio = preprocess_audio(audio_np)

    sr = runtime_config["sample_rate"]
    inputs = processor(audio=processed_audio, sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to(device=device, dtype=dtype) if v.dtype.is_floating_point else v.to(device=device)
              for k, v in inputs.items()}

    # Run encoder once, reuse for all target languages
    encoder_kwargs = {k: v for k, v in inputs.items() if k in ('input_features', 'attention_mask')}
    results = {}
    with torch.no_grad():
        t_enc = time.time()
        encoder_out = model.get_encoder()(**encoder_kwargs)
        logger.debug(f"Encoder: {(time.time() - t_enc)*1000:.0f}ms")

        for lang in target_langs:
            try:
                t_dec = time.time()
                output_tokens = model.generate(
                    **inputs,
                    encoder_outputs=encoder_out,
                    tgt_lang=lang,
                )
                logger.debug(f"Decoder [{lang}]: {(time.time() - t_dec)*1000:.0f}ms")
                text = processor.decode(output_tokens[0].tolist(), skip_special_tokens=True).strip()
                text = text.replace("#err", "")
                if text:
                    results[lang] = text
            except Exception as e:
                logger.error(f"Translation error for {lang}: {e}")
    return results

# --- SESSION MANAGER ---
@dataclass
class ClientSession:
    session_id: str
    websocket: WebSocket
    target_lang: str = field(default_factory=lambda: runtime_config["default_target_lang"])

class SessionManager:
    def __init__(self):
        self._sessions: dict[str, ClientSession] = {}
        self._lock = threading.Lock()

    async def connect(self, websocket: WebSocket) -> str:
        await websocket.accept()
        session_id = str(uuid.uuid4())
        session = ClientSession(session_id=session_id, websocket=websocket)
        with self._lock:
            self._sessions[session_id] = session
        return session_id

    def disconnect(self, session_id: str):
        with self._lock:
            self._sessions.pop(session_id, None)

    def set_language(self, session_id: str, lang: str):
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.target_lang = lang

    def get_unique_languages(self) -> set[str]:
        with self._lock:
            return {s.target_lang for s in self._sessions.values()}

    def get_sessions_for_language(self, lang: str) -> list[ClientSession]:
        with self._lock:
            return [s for s in self._sessions.values() if s.target_lang == lang]

    async def send_to_language(self, lang: str, message: str):
        sessions = self.get_sessions_for_language(lang)
        payload = json.dumps({"type": "subtitle", "text": message})

        async def _safe_send(session):
            try:
                await session.websocket.send_text(payload)
            except Exception:
                pass

        await asyncio.gather(*[_safe_send(s) for s in sessions])

    def client_count(self) -> int:
        with self._lock:
            return len(self._sessions)

manager = SessionManager()

# --- AUDIO STREAM MANAGEMENT ---
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

def audio_callback(indata, frames, time_info, status):
    if status:
        logger.warning(f"Audio status: {status}")
    audio_queue.put(indata.copy())

def restart_audio_stream(device_index=None, channel=0):
    """Restart audio stream with new device/channel settings."""
    global audio_stream, native_sample_rate, resampler
    with audio_stream_lock:
        if audio_stream is not None:
            try:
                audio_stream.stop()
                audio_stream.close()
                logger.info("Previous audio stream stopped.")
            except Exception as e:
                logger.warning(f"Error stopping audio stream: {e}")
            audio_stream = None

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

        native_sample_rate = native_sr

        # Create resampler if needed (native -> 16kHz)
        target_sr = runtime_config["sample_rate"]
        resampler = create_resampler(native_sr, target_sr)
        if resampler is not None:
            logger.info(f"Resampling active: {native_sr} Hz -> {target_sr} Hz")
        else:
            logger.info(f"No resampling needed: device already at {native_sr} Hz")

        # If channel > 0, we need to open enough channels
        if channel > 0:
            channels_to_open = max(channels_to_open, channel + 1)

        def multi_channel_callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"Audio status: {status}")
            # Extract the specific channel
            ch = runtime_config["audio_channel"]
            if ch < indata.shape[1]:
                audio_queue.put(indata[:, ch:ch+1].copy())
            else:
                audio_queue.put(indata[:, 0:1].copy())

        try:
            if channel > 0 or (device_index is not None and channels_to_open > 1):
                audio_stream = sd.InputStream(
                    device=device_index,
                    channels=channels_to_open,
                    samplerate=native_sr,
                    callback=multi_channel_callback,
                )
            else:
                audio_stream = sd.InputStream(
                    device=device_index,
                    channels=1,
                    samplerate=native_sr,
                    callback=audio_callback,
                )
            audio_stream.start()
            dev_name = "default" if device_index is None else sd.query_devices(device_index)['name']
            logger.info(f"Audio stream started: device={dev_name}, channel={channel}, native_sr={native_sr}")
        except Exception as e:
            logger.error(f"Failed to start audio stream: {e}")
            audio_stream = None
            raise

# --- PROCESSING LOOP ---
def submit_translation(audio_np, target_langs, loop):
    """Run translation in ThreadPoolExecutor and push results to translation_queue."""
    try:
        results = translate_audio(audio_np, list(target_langs))
        if results:
            asyncio.run_coroutine_threadsafe(translation_queue.put(results), loop)
    except Exception as e:
        logger.error(f"Translation Error: {e}")

def processing_loop(loop):
    MAX_BUFFER_SAMPLES = 48000 * 30  # max 30s @ 48kHz (safe upper bound)
    audio_buffer = np.zeros(MAX_BUFFER_SAMPLES, dtype=np.float32)
    buffer_pos = 0
    vad_buffer = torch.tensor([], dtype=torch.float32)
    prev_audio = np.array([], dtype=np.float32)
    silence_start = None
    is_speaking = False
    VAD_MIN_SAMPLES = 512  # Minimum samples needed for Silero VAD at 16kHz

    while not stop_event.is_set():
        try:
            chunk = audio_queue.get(timeout=0.1)
            chunk_np = np.concatenate(chunk).flatten()

            # Resample to 16kHz if needed
            chunk_np = resample_audio(chunk_np, resampler)

            # Compute audio level for VU meter
            global _audio_level_db
            level = compute_audio_level(chunk_np)
            with _audio_level_lock:
                _audio_level_db = level

            # Pre-allocated buffer: append chunk
            n = chunk_np.shape[0]
            if buffer_pos + n <= MAX_BUFFER_SAMPLES:
                audio_buffer[buffer_pos:buffer_pos + n] = chunk_np
                buffer_pos += n

            # VAD: accumulate as torch tensor (avoid repeated numpy->tensor conversion)
            chunk_tensor = torch.from_numpy(chunk_np).float()
            vad_buffer = torch.cat((vad_buffer, chunk_tensor))
            speech_detected = is_speaking  # default: keep previous state

            # Process all complete 512-sample frames in the VAD buffer
            while vad_buffer.shape[0] >= VAD_MIN_SAMPLES:
                vad_chunk = vad_buffer[:VAD_MIN_SAMPLES]
                vad_buffer = vad_buffer[VAD_MIN_SAMPLES:]
                confidence = vad_model(vad_chunk, runtime_config["sample_rate"]).item()
                speech_detected = confidence > 0.5

            sr = runtime_config["sample_rate"]  # 16kHz (after resampling)
            total_frames = buffer_pos  # O(1) instead of O(n)
            total_duration = total_frames / sr
            current_time = time.time()

            if speech_detected:
                is_speaking = True
                silence_start = None
            else:
                if silence_start is None:
                    silence_start = current_time

            # Read config values
            silence_dur = runtime_config["silence_duration"]
            max_chunk = runtime_config["max_chunk_duration"]
            min_chunk = runtime_config["min_chunk_duration"]
            ctx_overlap = runtime_config["context_overlap"]

            # Trigger transcription/translation
            should_process = False
            if is_speaking and silence_start and (current_time - silence_start > silence_dur):
                should_process = True
            if total_duration > max_chunk:
                should_process = True

            if should_process and total_duration >= min_chunk:
                full_audio = audio_buffer[:buffer_pos].copy().astype(np.float32)

                # Prepend context from previous segment
                audio_to_process = full_audio
                if prev_audio.size > 0:
                    audio_to_process = np.concatenate((prev_audio, full_audio))

                # Get unique languages from connected clients
                target_langs = manager.get_unique_languages()
                if target_langs:
                    # Backpressure: drop chunk if inference queue is already busy
                    if inference_executor._work_queue.qsize() >= 1:
                        logger.warning(f"Skipping chunk ({total_duration:.1f}s) — inference queue busy")
                    else:
                        inference_executor.submit(submit_translation, audio_to_process, target_langs, loop)

                # Save context overlap for next segment
                samples_ctx = int(sr * ctx_overlap)
                if full_audio.shape[0] > samples_ctx:
                    prev_audio = full_audio[-samples_ctx:]
                else:
                    prev_audio = full_audio

                buffer_pos = 0
                is_speaking = False
                silence_start = None

            # Cleanup long silence
            elif not is_speaking and total_duration > min_chunk:
                samples_to_keep = int(sr * 1.0)
                if total_frames > samples_to_keep * 5:
                    buffer_pos = 0

        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Loop Error: {e}")

# --- WEB SERVER ---
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request})

# --- API ENDPOINTS ---

@app.get("/api/status")
async def api_status():
    # Audio stream status
    with audio_stream_lock:
        if audio_stream is not None and audio_stream.active:
            dev_idx = runtime_config["audio_device_index"]
            dev_name = "default" if dev_idx is None else sd.query_devices(dev_idx)['name']
            audio_status = {
                "status": "running",
                "device_name": dev_name,
                "channel": runtime_config["audio_channel"],
                "native_sample_rate": native_sample_rate,
                "resampling_active": resampler is not None,
            }
        else:
            audio_status = {"status": "stopped", "device_name": None, "channel": 0}

    # GPU info
    gpu_name = None
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)

    with _audio_level_lock:
        current_audio_level = _audio_level_db

    return JSONResponse({
        "status": "ok",
        "clients": manager.client_count(),
        "audio_level_db": round(current_audio_level, 1),
        "active_languages": sorted(manager.get_unique_languages()),
        "device": device.type,
        "uptime": int(time.time() - start_time),
        "components": {
            "model": {"status": "running", "name": MODEL_NAME, "device": device.type, "gpu_name": gpu_name},
            "vad": {"status": "running", "type": "silero"},
            "audio_stream": audio_status,
            "inference_executor": {
                "status": "running",
                "pending_tasks": inference_executor._work_queue.qsize(),
            },
        },
        "config": dict(runtime_config),
    })

@app.get("/api/devices")
async def api_devices():
    try:
        devices = get_audio_devices()
        return JSONResponse({
            "devices": devices,
            "current_device_index": runtime_config["audio_device_index"],
            "current_channel": runtime_config["audio_channel"],
        })
    except Exception as e:
        logger.error(f"Error listing devices: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/devices/select")
async def api_devices_select(request: Request):
    try:
        body = await request.json()
        device_index = body.get("device_index")
        channel = body.get("channel", 0)

        # Validate device
        if device_index is not None:
            devices = sd.query_devices()
            if device_index < 0 or device_index >= len(devices):
                return JSONResponse({"error": f"Invalid device_index: {device_index}"}, status_code=400)
            dev = devices[device_index]
            if dev['max_input_channels'] <= 0:
                return JSONResponse({"error": f"Device {device_index} has no input channels"}, status_code=400)
            if channel < 0 or channel >= dev['max_input_channels']:
                return JSONResponse({"error": f"Channel {channel} out of range (0-{dev['max_input_channels']-1})"}, status_code=400)

        # Update config
        runtime_config["audio_device_index"] = device_index
        runtime_config["audio_channel"] = channel

        # Restart stream
        restart_audio_stream(device_index, channel)
        return JSONResponse({"ok": True})
    except Exception as e:
        logger.error(f"Error selecting device: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/config")
async def api_config_get():
    return JSONResponse(dict(runtime_config))

@app.post("/api/config")
async def api_config_post(request: Request):
    try:
        body = await request.json()
        errors = {}

        for key, value in body.items():
            if key not in runtime_config:
                errors[key] = f"Unknown config key: {key}"
                continue

            # Type check
            expected_type = type(DEFAULT_CONFIG[key])
            if DEFAULT_CONFIG[key] is None:
                # audio_device_index can be None or int
                if value is not None and not isinstance(value, int):
                    errors[key] = f"Expected int or null for {key}"
                    continue
            elif expected_type == bool:
                if not isinstance(value, bool):
                    errors[key] = f"Expected bool for {key}"
                    continue
            elif expected_type == float:
                if not isinstance(value, (int, float)):
                    errors[key] = f"Expected number for {key}"
                    continue
                value = float(value)
            elif expected_type == int:
                if not isinstance(value, (int,)) or isinstance(value, bool):
                    errors[key] = f"Expected int for {key}"
                    continue
            elif expected_type == str:
                if not isinstance(value, str):
                    errors[key] = f"Expected string for {key}"
                    continue

            # Range check
            if key in CONFIG_RANGES:
                min_val, max_val = CONFIG_RANGES[key]
                if value < min_val or value > max_val:
                    errors[key] = f"{key} must be between {min_val} and {max_val}"
                    continue

            runtime_config[key] = value

        if errors:
            return JSONResponse({"errors": errors, "config": dict(runtime_config)}, status_code=400)

        logger.info(f"Config updated: {body}")
        return JSONResponse(dict(runtime_config))
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.websocket("/api/logs")
async def api_logs_ws(websocket: WebSocket):
    await websocket.accept()
    listener = log_handler.add_listener()
    try:
        # Send history
        history = log_handler.get_history()
        await websocket.send_text(json.dumps({"type": "history", "entries": history}))

        # Stream new entries
        while True:
            entry = await listener.get()
            await websocket.send_text(json.dumps({"type": "log", "entry": entry}))
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        log_handler.remove_listener(listener)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    session_id = await manager.connect(websocket)
    logger.info(f"Client connected: {session_id[:8]}")
    try:
        while True:
            msg = await websocket.receive_text()
            data = json.loads(msg)
            if data.get("type") == "set_lang" and "lang" in data:
                manager.set_language(session_id, data["lang"])
                logger.info(f"Session {session_id[:8]} language -> {data['lang']}")
    except WebSocketDisconnect:
        manager.disconnect(session_id)
        logger.info(f"Client disconnected: {session_id[:8]}")

async def broadcaster():
    """Send translated texts to clients per their language."""
    while True:
        results = await translation_queue.get()
        tasks = []
        for lang, text in results.items():
            logger.info(f"[{lang}] {text}")
            tasks.append(manager.send_to_language(lang, text))
        await asyncio.gather(*tasks)

# --- MAIN ---
if __name__ == "__main__":
    device_idx = runtime_config["audio_device_index"]

    restart_audio_stream(device_idx, runtime_config["audio_channel"])

    @app.on_event("startup")
    async def startup_event():
        loop = asyncio.get_running_loop()
        asyncio.create_task(broadcaster())
        processing_thread = threading.Thread(target=processing_loop, args=(loop,), daemon=True)
        processing_thread.start()

    logger.info("Starting Web Server on port 8888...")
    uvicorn.run(app, host="0.0.0.0", port=8888, log_level="info")
