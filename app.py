import asyncio
import threading
import queue
import time
import json
import uuid
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import sounddevice as sd
import torch
from transformers import SeamlessM4Tv2ForSpeechToText, AutoProcessor
from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# --- CONFIG ---
#DEVICE_NAME_KEYWORD = "<<--Spotify2StudioLive"
#DEVICE_NAME_KEYWORD = "-->>StudioLive2Stream"
SAMPLE_RATE = 16000
CHANNELS = 1
SILENCE_DURATION = 0.8
MIN_CHUNK_DURATION = 1.5
MAX_CHUNK_DURATION = 12.0
CONTEXT_OVERLAP = 0.5
DEFAULT_TARGET_LANG = "ces"

# --- GLOBAL STATE ---
audio_queue = queue.Queue()
translation_queue = asyncio.Queue()
stop_event = threading.Event()

# --- DEVICE DETECTION ---
def detect_device():
    """Auto-detect best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.float16
        print(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
        return device, dtype
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float16
        print("Using device: MPS (Apple Silicon)")
        return device, dtype
    print("WARNING: Using CPU - translation will be slow")
    device = torch.device("cpu")
    dtype = torch.float32
    return device, dtype

device, dtype = detect_device()

# --- MODEL (direct, not pipeline) ---
MODEL_NAME = "facebook/seamless-m4t-v2-large"
print(f"Loading model {MODEL_NAME}...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = SeamlessM4Tv2ForSpeechToText.from_pretrained(MODEL_NAME)
model = model.to(device=device, dtype=dtype)
model.eval()

# torch.compile on CUDA if available
if device.type == "cuda":
    try:
        model = torch.compile(model)
        print("Model compiled with torch.compile()")
    except Exception as e:
        print(f"torch.compile() not available: {e}")

print("Model loaded.")

# --- MODEL WARM-UP ---
def warmup_model():
    """Run dummy inference to eliminate cold-start delay."""
    print("Warming up model...")
    dummy_audio = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)  # 2s silence
    inputs = processor(audios=dummy_audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    inputs = {k: v.to(device=device, dtype=dtype) if v.dtype.is_floating_point else v.to(device=device)
              for k, v in inputs.items()}
    with torch.no_grad():
        model.generate(**inputs, tgt_lang="ces", generate_speech=False)
    print("Warm-up complete.")

warmup_model()

# --- SILERO VAD ---
print("Loading Silero VAD...")
vad_model, vad_utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    trust_repo=True
)
(get_speech_timestamps, _, read_audio, _, _) = vad_utils
print("Silero VAD loaded.")

def is_speech(audio_chunk_np: np.ndarray) -> bool:
    """Check if audio chunk contains speech using Silero VAD."""
    tensor = torch.from_numpy(audio_chunk_np).float()
    if tensor.dim() > 1:
        tensor = tensor.squeeze()
    # Silero expects 16kHz mono
    confidence = vad_model(tensor, SAMPLE_RATE).item()
    return confidence > 0.5

# --- MULTI-LANGUAGE TRANSLATION ---
inference_executor = ThreadPoolExecutor(max_workers=1)

def translate_audio(audio_np: np.ndarray, target_langs: list[str]) -> dict[str, str]:
    """Translate audio to multiple languages. Encoder runs once, decoder per language."""
    inputs = processor(audios=audio_np, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    inputs = {k: v.to(device=device, dtype=dtype) if v.dtype.is_floating_point else v.to(device=device)
              for k, v in inputs.items()}

    results = {}
    with torch.no_grad():
        for lang in target_langs:
            try:
                output_tokens = model.generate(**inputs, tgt_lang=lang, generate_speech=False)
                text = processor.decode(output_tokens[0].tolist(), skip_special_tokens=True).strip()
                # Cleanup artifacts
                text = text.replace("#err", "")
                if text:
                    results[lang] = text
            except Exception as e:
                print(f"Translation error for {lang}: {e}")
    return results

# --- SESSION MANAGER ---
@dataclass
class ClientSession:
    session_id: str
    websocket: WebSocket
    target_lang: str = DEFAULT_TARGET_LANG

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
        for session in sessions:
            try:
                await session.websocket.send_text(payload)
            except Exception:
                pass

    def client_count(self) -> int:
        with self._lock:
            return len(self._sessions)

manager = SessionManager()

# --- AUDIO CALLBACK ---
def find_device_index(keyword):
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if keyword.lower() in dev['name'].lower() and dev['max_input_channels'] > 0:
            print(f"Found device '{dev['name']}' at index {i}")
            return i
    print("Device not found. Using default input device.")
    return None

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"Audio status: {status}")
    audio_queue.put(indata.copy())

# --- PROCESSING LOOP ---
def submit_translation(audio_np, target_langs, loop):
    """Run translation in ThreadPoolExecutor and push results to translation_queue."""
    try:
        results = translate_audio(audio_np, list(target_langs))
        if results:
            asyncio.run_coroutine_threadsafe(translation_queue.put(results), loop)
    except Exception as e:
        print(f"Translation Error: {e}")

def processing_loop(loop):
    buffer = []
    prev_audio = np.array([], dtype=np.float32)
    silence_start = None
    is_speaking = False

    while not stop_event.is_set():
        try:
            chunk = audio_queue.get(timeout=0.1)
            buffer.append(chunk)
            chunk_np = np.concatenate(chunk).flatten()

            # Silero VAD
            speech_detected = is_speech(chunk_np)

            total_frames = sum(c.shape[0] for c in buffer)
            total_duration = total_frames / SAMPLE_RATE
            current_time = time.time()

            if speech_detected:
                is_speaking = True
                silence_start = None
            else:
                if silence_start is None:
                    silence_start = current_time

            # Trigger transcription/translation
            should_process = False
            if is_speaking and silence_start and (current_time - silence_start > SILENCE_DURATION):
                should_process = True
            if total_duration > MAX_CHUNK_DURATION:
                should_process = True

            if should_process and total_duration >= MIN_CHUNK_DURATION:
                full_audio = np.concatenate([c.flatten() for c in buffer]).astype(np.float32)

                # Prepend context from previous segment
                audio_to_process = full_audio
                if prev_audio.size > 0:
                    audio_to_process = np.concatenate((prev_audio, full_audio))

                # Get unique languages from connected clients
                target_langs = manager.get_unique_languages()
                if target_langs:
                    # Submit to ThreadPoolExecutor (non-blocking)
                    inference_executor.submit(submit_translation, audio_to_process, target_langs, loop)

                # Save context overlap for next segment
                samples_ctx = int(SAMPLE_RATE * CONTEXT_OVERLAP)
                if full_audio.shape[0] > samples_ctx:
                    prev_audio = full_audio[-samples_ctx:]
                else:
                    prev_audio = full_audio

                buffer = []
                is_speaking = False
                silence_start = None

            # Cleanup long silence
            elif not is_speaking and total_duration > MIN_CHUNK_DURATION:
                samples_to_keep = int(SAMPLE_RATE * 1.0)
                if total_frames > samples_to_keep * 5:
                    buffer = []

        except queue.Empty:
            continue
        except Exception as e:
            print(f"Loop Error: {e}")

# --- WEB SERVER ---
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/status")
async def api_status():
    return JSONResponse({
        "status": "ok",
        "clients": manager.client_count(),
        "active_languages": sorted(manager.get_unique_languages()),
        "device": device.type,
    })

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    session_id = await manager.connect(websocket)
    print(f"Client connected: {session_id}")
    try:
        while True:
            msg = await websocket.receive_text()
            data = json.loads(msg)
            if data.get("type") == "set_lang" and "lang" in data:
                manager.set_language(session_id, data["lang"])
                print(f"Session {session_id[:8]} language → {data['lang']}")
    except WebSocketDisconnect:
        manager.disconnect(session_id)
        print(f"Client disconnected: {session_id[:8]}")

async def broadcaster():
    """Send translated texts to clients per their language."""
    while True:
        results = await translation_queue.get()  # dict[str, str]
        for lang, text in results.items():
            print(f"[{lang}] {text}")
            await manager.send_to_language(lang, text)

# --- MAIN ---
if __name__ == "__main__":
    device_idx = find_device_index(DEVICE_NAME_KEYWORD)

    audio_stream = sd.InputStream(
        device=device_idx,
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        callback=audio_callback
    )
    audio_stream.start()
    print("Audio stream started.")

    @app.on_event("startup")
    async def startup_event():
        loop = asyncio.get_running_loop()
        asyncio.create_task(broadcaster())
        processing_thread = threading.Thread(target=processing_loop, args=(loop,), daemon=True)
        processing_thread.start()

    print("Starting Web Server on port 8888...")
    uvicorn.run(app, host="0.0.0.0", port=8888, log_level="info")
