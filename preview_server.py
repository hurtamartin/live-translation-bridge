"""Lightweight preview server - serves only the frontend without ML/audio dependencies."""
import asyncio
import base64
import collections
import json
import logging
import math
import os
import random
import secrets
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

app = FastAPI()

# --- CORS ---
cors_origins = os.environ.get("CORS_ORIGINS", "").strip()
if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in cors_origins.split(",")],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

DEFAULT_TARGET_LANG = "ces"
SUPPORTED_LANGUAGES = {"ces", "eng", "rus", "ukr", "deu", "spa"}
MAX_CLIENTS = 200
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

# --- Config persistence ---
CONFIG_FILE = Path(__file__).parent / "config.json"

DEFAULT_CONFIG = {
    "audio_device_index": None,
    "audio_channel": 0,
    "sample_rate": 16000,
    "silence_duration": 0.8,
    "min_chunk_duration": 1.5,
    "max_chunk_duration": 12.0,
    "context_overlap": 0.5,
    "default_target_lang": "ces",
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
    "sample_rate": (8000, 48000),
    "audio_channel": (0, 127),
    "preprocess_noise_gate_threshold": (-60.0, -10.0),
    "preprocess_normalize_target": (-20.0, 0.0),
    "preprocess_highpass_cutoff": (20, 300),
}

def load_config() -> dict:
    config = dict(DEFAULT_CONFIG)
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            for key, value in saved.items():
                if key in DEFAULT_CONFIG:
                    config[key] = value
        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using defaults")
    return config

def save_config():
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(runtime_config, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to save config: {e}")

runtime_config = load_config()

# --- Simple in-memory rate limiter ---
_rate_limit_store: dict[str, list[float]] = {}
_rate_limit_lock = threading.Lock()
RATE_LIMIT_MAX = int(os.environ.get("RATE_LIMIT_MAX", "10"))
RATE_LIMIT_WINDOW = 1.0


def _check_rate_limit(client_ip: str) -> bool:
    now = time.time()
    with _rate_limit_lock:
        timestamps = _rate_limit_store.get(client_ip, [])
        timestamps = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW]
        if len(timestamps) >= RATE_LIMIT_MAX:
            _rate_limit_store[client_ip] = timestamps
            return False
        timestamps.append(now)
        _rate_limit_store[client_ip] = timestamps
        return True


def rate_limit(request: Request):
    client_ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Too many requests")


# --- Mock audio level & metrics ---
_audio_level_db = -60.0
_audio_level_peak = -60.0
_audio_history: collections.deque = collections.deque(maxlen=60)

_perf_metrics = {
    "encoder_ms": collections.deque(maxlen=100),
    "decoder_ms": collections.deque(maxlen=100),
    "total_translations": 0,
    "last_inference_time": None,
}

_translation_history: collections.deque = collections.deque(maxlen=50)

def _update_audio_history():
    global _audio_level_db, _audio_level_peak
    # Mock: simulate fluctuating audio level
    _audio_level_db = -30 + 15 * math.sin(time.time() * 0.5) + random.uniform(-3, 3)
    _audio_level_peak = max(_audio_level_peak - 0.5, _audio_level_db)
    _audio_history.append({
        "time": time.time(),
        "rms_db": round(_audio_level_db, 1),
        "peak_db": round(_audio_level_peak, 1),
    })

# --- Demo data ---
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
    "rus": [
        "\u0414\u043E\u0431\u0440\u043E \u043F\u043E\u0436\u0430\u043B\u043E\u0432\u0430\u0442\u044C \u043D\u0430 \u0441\u0435\u0433\u043E\u0434\u043D\u044F\u0448\u043D\u0435\u0435 \u0441\u043E\u0431\u0440\u0430\u043D\u0438\u0435.",
        "\u042D\u0442\u043E \u043F\u0440\u0438\u043C\u0435\u0440 \u043F\u0435\u0440\u0435\u0432\u0435\u0434\u0451\u043D\u043D\u043E\u0433\u043E \u0442\u0435\u043A\u0441\u0442\u0430.",
        "\u0421\u0435\u0433\u043E\u0434\u043D\u044F \u043C\u044B \u0431\u0443\u0434\u0435\u043C \u0433\u043E\u0432\u043E\u0440\u0438\u0442\u044C \u043E \u0432\u0430\u0436\u043D\u043E\u0439 \u0442\u0435\u043C\u0435.",
        "\u0422\u0430\u043A \u0431\u0443\u0434\u0443\u0442 \u0432\u044B\u0433\u043B\u044F\u0434\u0435\u0442\u044C \u0441\u0443\u0431\u0442\u0438\u0442\u0440\u044B.",
        "\u041F\u043E\u0436\u0430\u043B\u0443\u0439\u0441\u0442\u0430, \u0432\u043A\u043B\u044E\u0447\u0438\u0442\u0435 \u0441\u0443\u0431\u0442\u0438\u0442\u0440\u044B \u043D\u0430 \u0441\u0432\u043E\u0451\u043C \u044F\u0437\u044B\u043A\u0435.",
        "\u0421\u043F\u0430\u0441\u0438\u0431\u043E \u0437\u0430 \u0432\u0430\u0448\u0435 \u0443\u0447\u0430\u0441\u0442\u0438\u0435.",
        "\u0422\u0435\u043F\u0435\u0440\u044C \u043F\u0435\u0440\u0435\u0439\u0434\u0451\u043C \u043A \u0433\u043B\u0430\u0432\u043D\u043E\u043C\u0443 \u043F\u0443\u043D\u043A\u0442\u0443 \u043F\u043E\u0432\u0435\u0441\u0442\u043A\u0438 \u0434\u043D\u044F.",
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
    "spa": [
        "Bienvenidos a la reuni\u00f3n de hoy.",
        "Este es un ejemplo de texto traducido.",
        "Hoy hablaremos sobre un tema importante.",
        "As\u00ed se ver\u00e1n los subt\u00edtulos.",
        "Por favor, active los subt\u00edtulos en su idioma.",
        "Gracias por su participaci\u00f3n.",
        "Ahora pasamos al punto principal de la agenda.",
    ],
}

@dataclass
class ClientSession:
    session_id: str
    websocket: WebSocket
    target_lang: str = DEFAULT_TARGET_LANG
    connected_at: float = field(default_factory=time.time)
    client_ip: str = ""

sessions: dict[str, ClientSession] = {}

# --- ADMIN AUTH ---
security = HTTPBasic()
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin")

if ADMIN_USERNAME == "admin" and ADMIN_PASSWORD == "admin":
    logger.warning("Using default admin credentials — set ADMIN_USERNAME and ADMIN_PASSWORD env vars for production")


def verify_admin(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


def _verify_ws_auth(websocket: WebSocket) -> bool:
    """Verify admin credentials for WebSocket connections.

    Checks Authorization header first (works for non-browser clients),
    then falls back to ?token= query parameter (Base64-encoded user:pass)
    because browser WebSocket API does not support custom headers.
    """
    # Try Authorization header first
    auth_header = websocket.headers.get("authorization", "")
    if auth_header.startswith("Basic "):
        try:
            decoded = base64.b64decode(auth_header[6:]).decode("utf-8")
            username, password = decoded.split(":", 1)
            if secrets.compare_digest(username, ADMIN_USERNAME) and secrets.compare_digest(password, ADMIN_PASSWORD):
                return True
        except Exception:
            pass
    # Fallback: ?token=base64(user:pass) for browser WebSocket
    token = websocket.query_params.get("token", "")
    if token:
        try:
            decoded = base64.b64decode(token).decode("utf-8")
            username, password = decoded.split(":", 1)
            if secrets.compare_digest(username, ADMIN_USERNAME) and secrets.compare_digest(password, ADMIN_PASSWORD):
                return True
        except Exception:
            pass
    return False


# --- Config validation (same as main server) ---

def _validate_config_values(body: dict) -> tuple[dict, dict]:
    """Validate config key/value pairs. Returns (validated, errors) dicts."""
    validated = {}
    errors = {}

    for key, value in body.items():
        if key not in DEFAULT_CONFIG:
            errors[key] = f"Unknown config key: {key}"
            continue

        expected_type = type(DEFAULT_CONFIG[key])
        if DEFAULT_CONFIG[key] is None:
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

        if key in CONFIG_RANGES:
            min_val, max_val = CONFIG_RANGES[key]
            if value < min_val or value > max_val:
                errors[key] = f"{key} must be between {min_val} and {max_val}"
                continue

        validated[key] = value

    return validated, errors


# --- ROUTES ---

@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request, credentials: HTTPBasicCredentials = Depends(security)):
    if not (secrets.compare_digest(credentials.username, ADMIN_USERNAME) and
            secrets.compare_digest(credentials.password, ADMIN_PASSWORD)):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    ws_token = base64.b64encode(f"{credentials.username}:{credentials.password}".encode()).decode()
    return templates.TemplateResponse("admin.html", {"request": request, "ws_token": ws_token})

@app.get("/api/status")
async def api_status(_user: str = Depends(verify_admin), _rl=Depends(rate_limit)):
    return JSONResponse({
        "status": "ok",
        "clients": len(sessions),
        "audio_level_db": round(_audio_level_db, 1),
        "audio_level_peak": round(_audio_level_peak, 1),
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
async def api_devices(_user: str = Depends(verify_admin), _rl=Depends(rate_limit)):
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
async def api_devices_select(request: Request, _user: str = Depends(verify_admin), _rl=Depends(rate_limit)):
    try:
        body = await request.json()
        device_index = body.get("device_index")
        channel = body.get("channel", 0)
        runtime_config["audio_device_index"] = device_index
        runtime_config["audio_channel"] = channel
        save_config()
        logger.info(f"Device selected: index={device_index}, channel={channel} (mock)")
        return JSONResponse({"ok": True})
    except Exception as e:
        logger.error(f"Error selecting device: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/config")
async def api_config_get(_user: str = Depends(verify_admin), _rl=Depends(rate_limit)):
    return JSONResponse(dict(runtime_config))

@app.post("/api/config")
async def api_config_post(request: Request, _user: str = Depends(verify_admin), _rl=Depends(rate_limit)):
    try:
        body = await request.json()
        validated, errors = _validate_config_values(body)

        for key, value in validated.items():
            runtime_config[key] = value

        if errors:
            return JSONResponse({"errors": errors, "config": dict(runtime_config)}, status_code=400)

        save_config()
        logger.info(f"Config updated: {body}")
        return JSONResponse(dict(runtime_config))
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

# --- Config export/import ---

@app.get("/api/config/export")
async def api_config_export(_user: str = Depends(verify_admin), _rl=Depends(rate_limit)):
    return JSONResponse(dict(runtime_config))

@app.post("/api/config/import")
async def api_config_import(request: Request, _user: str = Depends(verify_admin), _rl=Depends(rate_limit)):
    try:
        body = await request.json()
        validated, errors = _validate_config_values(body)

        for key, value in validated.items():
            runtime_config[key] = value

        if validated:
            save_config()

        logger.info(f"Config imported: {len(validated)} keys" + (f", {len(errors)} rejected" if errors else ""))
        result = {"ok": True, "imported": len(validated), "config": dict(runtime_config)}
        if errors:
            result["errors"] = errors
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

# --- Health endpoint (no auth) ---

@app.get("/health")
async def health_check():
    return JSONResponse({
        "status": "healthy",
        "uptime": int(time.time() - start_time),
        "clients": len(sessions),
        "audio_stream": "running",
        "model": "loaded",
        "processing_thread": "running",
        "gpu": "ok",
    })

# --- Sessions, metrics, translation history ---

@app.get("/api/qr.svg")
async def qr_code_svg(request: Request):
    """Generate QR code SVG dynamically from the request URL."""
    import qrcode
    import qrcode.image.svg
    import io

    base_url = str(request.base_url).rstrip("/")
    img = qrcode.make(base_url, image_factory=qrcode.image.svg.SvgPathImage)
    buf = io.BytesIO()
    img.save(buf)
    return Response(content=buf.getvalue(), media_type="image/svg+xml",
                    headers={"Cache-Control": "no-cache"})


@app.get("/api/sessions")
async def api_sessions(_user: str = Depends(verify_admin), _rl=Depends(rate_limit)):
    now = time.time()
    result = []
    for s in sessions.values():
        result.append({
            "id": s.session_id[:8],
            "lang": s.target_lang,
            "ip": s.client_ip,
            "connected_for": int(now - s.connected_at),
        })
    return JSONResponse(result)

@app.get("/api/metrics")
async def api_metrics(_user: str = Depends(verify_admin), _rl=Depends(rate_limit)):
    enc = list(_perf_metrics["encoder_ms"])
    dec = list(_perf_metrics["decoder_ms"])
    total = _perf_metrics["total_translations"]
    last_time = _perf_metrics["last_inference_time"]
    return JSONResponse({
        "total_translations": total,
        "avg_encoder_ms": round(sum(enc) / len(enc), 1) if enc else 0,
        "avg_decoder_ms": round(sum(dec) / len(dec), 1) if dec else 0,
        "last_encoder_ms": round(enc[-1], 1) if enc else 0,
        "last_decoder_ms": round(dec[-1], 1) if dec else 0,
        "last_inference_ago": round(time.time() - last_time, 1) if last_time else None,
        "gpu_name": None,
        "gpu_memory_used_mb": None,
        "gpu_memory_total_mb": None,
    })

@app.get("/api/translations")
async def api_translations(_user: str = Depends(verify_admin), _rl=Depends(rate_limit)):
    return JSONResponse(list(_translation_history))

# --- Audio history ---

@app.get("/api/audio-history")
async def api_audio_history(_user: str = Depends(verify_admin), _rl=Depends(rate_limit)):
    return JSONResponse(list(_audio_history))

@app.get("/api/audio-level")
async def api_audio_level(_user: str = Depends(verify_admin), _rl=Depends(rate_limit)):
    return JSONResponse({"rms_db": round(_audio_level_db, 1)})

# --- Status WebSocket ---

@app.websocket("/api/status/ws")
async def api_status_ws(websocket: WebSocket):
    global _audio_level_peak
    if not _verify_ws_auth(websocket):
        await websocket.close(code=4401)
        return
    await websocket.accept()
    try:
        while True:
            payload = {
                "clients": len(sessions),
                "audio_level_db": round(_audio_level_db, 1),
                "audio_level_peak": round(_audio_level_peak, 1),
                "active_languages": sorted({s.target_lang for s in sessions.values()}),
                "device": "cpu",
                "uptime": int(time.time() - start_time),
                "components": {
                    "model": {"status": "running", "name": "preview-mock", "device": "cpu", "gpu_name": None},
                    "vad": {"status": "running", "type": "mock"},
                    "audio_stream": {"status": "running", "device_name": "Preview (mock)", "channel": 0},
                    "inference_executor": {"status": "running", "pending_tasks": 0},
                },
            }
            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass

# --- Log WebSocket ---

@app.websocket("/api/logs")
async def api_logs_ws(websocket: WebSocket):
    if not _verify_ws_auth(websocket):
        await websocket.close(code=4401)
        return
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

# --- Client WebSocket ---

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    if len(sessions) >= MAX_CLIENTS:
        await websocket.close(code=1013, reason="Too many clients")
        logger.warning("Client rejected: connection limit reached")
        return
    await websocket.accept()
    session_id = str(uuid.uuid4())
    client_ip = websocket.client.host if websocket.client else "unknown"
    sessions[session_id] = ClientSession(
        session_id=session_id,
        websocket=websocket,
        client_ip=client_ip,
    )
    logger.info(f"Client connected: {session_id[:8]}")
    try:
        while True:
            msg = await websocket.receive_text()
            try:
                data = json.loads(msg)
            except (json.JSONDecodeError, ValueError):
                logger.warning(f"Session {session_id[:8]} sent invalid JSON")
                continue
            msg_type = data.get("type")
            if msg_type == "ping":
                await websocket.send_text('{"type":"pong"}')
            elif msg_type == "set_lang" and "lang" in data:
                lang = data["lang"]
                if not isinstance(lang, str) or lang not in SUPPORTED_LANGUAGES:
                    logger.warning(f"Session {session_id[:8]} invalid language: {str(lang)[:20]}")
                    continue
                sessions[session_id].target_lang = lang
                logger.info(f"Session {session_id[:8]} language -> {lang}")
    except WebSocketDisconnect:
        pass
    finally:
        sessions.pop(session_id, None)
        logger.info(f"Client disconnected: {session_id[:8]}")

# --- Background tasks ---

async def demo_subtitles():
    """Send demo subtitles every few seconds, per-session language."""
    idx = 0
    await asyncio.sleep(3)
    while True:
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

        # Mock: add to translation history
        if lang_sessions:
            _perf_metrics["total_translations"] += 1
            _perf_metrics["last_inference_time"] = time.time()
            _perf_metrics["encoder_ms"].append(random.uniform(20, 80))
            _perf_metrics["decoder_ms"].append(random.uniform(50, 200))
            for lang in lang_sessions:
                demos = DEMOS.get(lang, DEMOS["ces"])
                _translation_history.appendleft({
                    "timestamp": time.time(),
                    "translations": {lang: demos[idx % len(demos)]},
                })

        idx += 1
        await asyncio.sleep(4)

async def audio_history_ticker():
    while True:
        _update_audio_history()
        await asyncio.sleep(1)

@app.on_event("startup")
async def startup():
    asyncio.create_task(demo_subtitles())
    asyncio.create_task(audio_history_ticker())

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8888"))
    logger.info(f"Preview server at http://localhost:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
