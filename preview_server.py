"""Lightweight preview server - serves only the frontend without ML/audio dependencies."""
import asyncio
import collections
import json
import logging
import math
import random
import time
import uuid
from dataclasses import dataclass
from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

DEFAULT_TARGET_LANG = "ces"
start_time = time.time()

# --- Logging with buffer (same as app.py) ---
logger = logging.getLogger("preview")
logger.setLevel(logging.DEBUG)

class LogBufferHandler(logging.Handler):
    def __init__(self, maxlen=500):
        super().__init__()
        self.buffer = collections.deque(maxlen=maxlen)
        self.listeners: list[asyncio.Queue] = []

    def emit(self, record):
        entry = {
            "time": time.strftime("%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "message": self.format(record),
        }
        self.buffer.append(entry)
        for q in self.listeners:
            try:
                q.put_nowait(entry)
            except asyncio.QueueFull:
                pass

    def add_listener(self) -> asyncio.Queue:
        q = asyncio.Queue(maxsize=100)
        self.listeners.append(q)
        return q

    def remove_listener(self, q: asyncio.Queue):
        try:
            self.listeners.remove(q)
        except ValueError:
            pass

    def get_history(self) -> list[dict]:
        return list(self.buffer)

log_handler = LogBufferHandler()
log_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(log_handler)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(console_handler)

# --- Mock runtime config ---
runtime_config = {
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
    "preprocess_noise_gate_threshold": -40.0,
    "preprocess_normalize": False,
    "preprocess_normalize_target": -3.0,
    "preprocess_highpass": False,
    "preprocess_highpass_cutoff": 80,
    "preprocess_auto_language": False,
}

CONFIG_RANGES = {
    "silence_duration": (0.3, 3.0),
    "min_chunk_duration": (0.5, 5.0),
    "max_chunk_duration": (5.0, 30.0),
    "context_overlap": (0.0, 2.0),
    "preprocess_noise_gate_threshold": (-60.0, -10.0),
    "preprocess_normalize_target": (-20.0, 0.0),
    "preprocess_highpass_cutoff": (20, 300),
}

DEMOS = {
    "ces": [
        "Vítejte na dnešním shromáždění.",
        "Ukázka přeloženého textu.",
        "Dnes budeme mluvit o důležitém tématu.",
        "Toto je ukázka, jak budou titulky vypadat.",
        "Prosím, zapněte si titulky ve svém jazyce.",
        "Děkujeme za vaši účast.",
        "Nyní přejdeme k hlavnímu bodu programu.",
    ],
    "eng": [
        "Welcome to today's gathering.",
        "This is a sample of translated text.",
        "Today we will talk about an important topic.",
        "This is how the subtitles will look.",
        "Please turn on subtitles in your language.",
        "Thank you for your participation.",
        "Now we move to the main point of the agenda.",
    ],
    "spa": [
        "Bienvenidos a la reunión de hoy.",
        "Este es un ejemplo de texto traducido.",
        "Hoy hablaremos sobre un tema importante.",
        "Así se verán los subtítulos.",
        "Por favor, active los subtítulos en su idioma.",
        "Gracias por su participación.",
        "Ahora pasamos al punto principal de la agenda.",
    ],
    "ukr": [
        "Ласкаво просимо на сьогоднішнє зібрання.",
        "Це приклад перекладеного тексту.",
        "Сьогодні ми говоритимемо про важливу тему.",
        "Ось як виглядатимуть субтитри.",
        "Будь ласка, увімкніть субтитри своєю мовою.",
        "Дякуємо за вашу участь.",
        "Тепер перейдемо до головного пункту порядку денного.",
    ],
    "deu": [
        "Willkommen bei der heutigen Versammlung.",
        "Dies ist ein Beispiel für übersetzten Text.",
        "Heute werden wir über ein wichtiges Thema sprechen.",
        "So werden die Untertitel aussehen.",
        "Bitte schalten Sie die Untertitel in Ihrer Sprache ein.",
        "Vielen Dank für Ihre Teilnahme.",
        "Jetzt kommen wir zum Hauptpunkt der Tagesordnung.",
    ],
    "pol": [
        "Witamy na dzisiejszym zgromadzeniu.",
        "To jest przykład przetłumaczonego tekstu.",
        "Dziś będziemy mówić o ważnym temacie.",
        "Tak będą wyglądać napisy.",
        "Proszę, włączcie napisy w swoim języku.",
        "Dziękujemy za udział.",
        "Teraz przechodzimy do głównego punktu porządku obrad.",
    ],
}

@dataclass
class ClientSession:
    session_id: str
    websocket: WebSocket
    target_lang: str = DEFAULT_TARGET_LANG

sessions: dict[str, ClientSession] = {}

@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request})

@app.get("/api/status")
async def api_status():
    return JSONResponse({
        "status": "ok",
        "clients": len(sessions),
        "active_languages": sorted({s.target_lang for s in sessions.values()}),
        "device": "preview",
        "uptime": int(time.time() - start_time),
        "components": {
            "model": {"status": "running", "name": "preview-mock", "device": "cpu", "gpu_name": None},
            "vad": {"status": "running", "type": "mock"},
            "audio_stream": {"status": "running", "device_name": "Preview (mock)", "channel": 0},
            "inference_executor": {"status": "running", "pending_tasks": 0},
        },
        "config": dict(runtime_config),
    })

@app.get("/api/devices")
async def api_devices():
    return JSONResponse({
        "devices": [
            {
                "index": 0,
                "name": "Preview Mock Device",
                "max_input_channels": 2,
                "default_samplerate": 48000,
                "is_default": True,
            },
            {
                "index": 1,
                "name": "Preview Multichannel (demo)",
                "max_input_channels": 16,
                "default_samplerate": 48000,
                "is_default": False,
            },
        ],
        "current_device_index": runtime_config["audio_device_index"],
        "current_channel": runtime_config["audio_channel"],
    })

@app.post("/api/devices/select")
async def api_devices_select(request: Request):
    body = await request.json()
    device_index = body.get("device_index")
    channel = body.get("channel", 0)
    runtime_config["audio_device_index"] = device_index
    runtime_config["audio_channel"] = channel
    logger.info(f"Device selected: index={device_index}, channel={channel} (mock)")
    return JSONResponse({"ok": True})

@app.get("/api/config")
async def api_config_get():
    return JSONResponse(dict(runtime_config))

@app.post("/api/config")
async def api_config_post(request: Request):
    body = await request.json()
    errors = {}
    for key, value in body.items():
        if key not in runtime_config:
            errors[key] = f"Unknown config key: {key}"
            continue
        if key in CONFIG_RANGES:
            min_val, max_val = CONFIG_RANGES[key]
            if not isinstance(value, (int, float)) or value < min_val or value > max_val:
                errors[key] = f"{key} must be between {min_val} and {max_val}"
                continue
        runtime_config[key] = value
    if errors:
        return JSONResponse({"errors": errors, "config": dict(runtime_config)}, status_code=400)
    logger.info(f"Config updated: {body}")
    return JSONResponse(dict(runtime_config))

@app.get("/api/audio-level")
async def api_audio_level():
    # Mock: simulate fluctuating audio level between -50 and -5 dB
    db = -30 + 15 * math.sin(time.time() * 0.5) + random.uniform(-3, 3)
    return JSONResponse({"rms_db": round(db, 1)})

@app.websocket("/api/logs")
async def api_logs_ws(websocket: WebSocket):
    await websocket.accept()
    listener = log_handler.add_listener()
    try:
        history = log_handler.get_history()
        await websocket.send_text(json.dumps({"type": "history", "entries": history}))
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
    await websocket.accept()
    session_id = str(uuid.uuid4())
    sessions[session_id] = ClientSession(session_id=session_id, websocket=websocket)
    logger.info(f"Client connected: {session_id[:8]}")
    try:
        while True:
            msg = await websocket.receive_text()
            data = json.loads(msg)
            if data.get("type") == "set_lang" and "lang" in data:
                sessions[session_id].target_lang = data["lang"]
                logger.info(f"Session {session_id[:8]} language -> {data['lang']}")
    except WebSocketDisconnect:
        sessions.pop(session_id, None)
        logger.info(f"Client disconnected: {session_id[:8]}")

async def demo_subtitles():
    """Send demo subtitles every few seconds, per-session language."""
    idx = 0
    await asyncio.sleep(3)
    while True:
        # Group sessions by language
        lang_sessions: dict[str, list[ClientSession]] = {}
        for s in list(sessions.values()):
            lang_sessions.setdefault(s.target_lang, []).append(s)

        for lang, lang_clients in lang_sessions.items():
            demos = DEMOS.get(lang, DEMOS["ces"])
            text = demos[idx % len(demos)]
            payload = json.dumps({"type": "subtitle", "text": text})
            for session in lang_clients:
                try:
                    await session.websocket.send_text(payload)
                except Exception:
                    pass
        idx += 1
        await asyncio.sleep(4)

@app.on_event("startup")
async def startup():
    asyncio.create_task(demo_subtitles())

if __name__ == "__main__":
    logger.info("Preview server at http://localhost:8888")
    uvicorn.run(app, host="0.0.0.0", port=8888, log_level="info")
