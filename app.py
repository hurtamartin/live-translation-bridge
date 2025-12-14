import asyncio
import threading
import queue
import time
import json
import numpy as np
import sounddevice as sd
import torch
from transformers import pipeline
from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn

# --- CONFIG ---
DEVICE_NAME_KEYWORD = "<<--Spotify2StudioLive"
DEVICE_NAME_KEYWORD = "-->>StudioLive2Stream"
SAMPLE_RATE = 16000
CHANNELS = 1
SILENCE_THRESHOLD = 0.01
SILENCE_DURATION = 1.0
MIN_CHUNK_DURATION = 2.0
MAX_CHUNK_DURATION = 15.0
DEFAULT_TARGET_LANG = "ces"

# --- GLOBAL STATE ---
audio_queue = queue.Queue()
text_queue = asyncio.Queue()
stop_event = threading.Event()
TARGET_LANG = DEFAULT_TARGET_LANG

# --- MODEL / PIPELINE ---
device = 0 if torch.backends.mps.is_available() else -1
print(f"Using device: {'MPS' if device==0 else 'CPU'}")

translator = pipeline(
    task="automatic-speech-recognition",
    model="facebook/seamless-m4t-v2-large",
    device=device,
    trust_remote_code=True
)
# Cast the underlying model to float16 if using MPS
if torch.backends.mps.is_available():
    translator.model = translator.model.to(torch.float16).to("mps")
    print("Model cast to FP16 on MPS")
    
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
            energy = np.sqrt(np.mean(chunk_np**2))

            total_frames = sum(c.shape[0] for c in buffer)
            total_duration = total_frames / SAMPLE_RATE
            current_time = time.time()

            # VAD
            if energy > SILENCE_THRESHOLD:
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

                try:
                    # Translate to TARGET_LANG
                    result = translator(audio_to_process, tgt_lang=TARGET_LANG)
                    translated_text = result["text"].strip()

                    # --- CLEANUP #err ---
                    translated_text = translated_text.replace("#err", "")

                    if translated_text:
                        print(f"[{TARGET_LANG}] {translated_text}")
                        asyncio.run_coroutine_threadsafe(text_queue.put(translated_text), loop)

                except Exception as e:
                    print(f"Transcription Error: {e}")

                # Save last 1s as context for next time
                samples_ctx = int(SAMPLE_RATE * 0.2)
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
                    # Also clear context if we had a massive silence, 
                    # though arguably keeping the "silence" as context is fine too.
                    # We'll stick to the request: persist context from "previous sentence".
                    # If we drop the buffer here, it wasn't a sentence.

        except queue.Empty:
            continue
        except Exception as e:
            print(f"Loop Error: {e}")

# --- WEB SERVER ---
app = FastAPI()
templates = Jinja2Templates(directory="templates")

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                pass

manager = ConnectionManager()

@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "target_lang": TARGET_LANG})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            msg = await websocket.receive_text()
            data = json.loads(msg)
            if data.get("type") == "set_lang" and "lang" in data:
                global TARGET_LANG
                TARGET_LANG = data["lang"]
                print(f"Target language changed to {TARGET_LANG}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)

async def broadcaster():
    while True:
        text = await text_queue.get()
        await manager.broadcast(text)

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
        # Start broadcaster task
        asyncio.create_task(broadcaster())
        # Start audio processing thread with correct loop
        processing_thread = threading.Thread(target=processing_loop, args=(loop,), daemon=True)
        processing_thread.start()

    print("Starting Web Server on port 8888...")
    uvicorn.run(app, host="0.0.0.0", port=8888, log_level="info")
