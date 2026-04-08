import asyncio
import base64
import json
import os
import queue
import threading
import time
from collections import OrderedDict, deque

import numpy as np
import secrets
import sounddevice as sd
import torch
from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.logging_handler import logger, log_handler
from src.config import (
    save_config, DEFAULT_CONFIG, CONFIG_RANGES, SUPPORTED_LANGUAGES,
    FIXED_CONFIG_VALUES, get_config_snapshot, apply_runtime_config_updates,
)

# --- Simple in-memory rate limiter (OrderedDict for O(1) eviction) ---
_rate_limit_store: OrderedDict[str, deque] = OrderedDict()
_rate_limit_lock = threading.Lock()
RATE_LIMIT_MAX = int(os.environ.get("RATE_LIMIT_MAX", "10"))  # requests per window
RATE_LIMIT_WINDOW = 1.0  # seconds
_RATE_LIMIT_MAX_ENTRIES = 5000
MAX_JSON_BODY_BYTES = int(os.environ.get("MAX_JSON_BODY_BYTES", "65536"))


def _check_rate_limit(client_ip: str) -> bool:
    """Return True if request is allowed, False if rate-limited."""
    now = time.time()
    with _rate_limit_lock:
        # Evict oldest entries when store grows too large (O(1) per eviction)
        while len(_rate_limit_store) > _RATE_LIMIT_MAX_ENTRIES:
            _rate_limit_store.popitem(last=False)

        timestamps = _rate_limit_store.get(client_ip)
        if timestamps is None:
            timestamps = deque(maxlen=RATE_LIMIT_MAX + 1)
            _rate_limit_store[client_ip] = timestamps
        else:
            # Move to end to maintain LRU order
            _rate_limit_store.move_to_end(client_ip)

        # Remove expired entries from front (deque is chronologically ordered)
        cutoff = now - RATE_LIMIT_WINDOW
        while timestamps and timestamps[0] < cutoff:
            timestamps.popleft()

        if len(timestamps) >= RATE_LIMIT_MAX:
            return False
        timestamps.append(now)
        return True


def rate_limit(request: Request):
    """FastAPI dependency for rate limiting admin API endpoints."""
    client_ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Too many requests")


async def _read_json_body(request: Request) -> dict:
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_JSON_BODY_BYTES:
                raise HTTPException(status_code=413, detail="Request body too large")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid Content-Length")
    body = await request.body()
    if len(body) > MAX_JSON_BODY_BYTES:
        raise HTTPException(status_code=413, detail="Request body too large")
    try:
        data = json.loads(body or b"{}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="Expected JSON object")
    return data
from src.translation.engine import MODEL_NAME, translate_audio
from src.audio.capture import (
    get_audio_devices, restart_audio_stream, compute_audio_level,
    resample_audio,
)
import src.state as state

# --- WEB SERVER ---
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

_CSP_VALUE = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; "
    "connect-src 'self' ws: wss:; "
    "font-src 'self'"
)


class CSPMiddleware:
    """Lightweight ASGI middleware — no per-request task/stream overhead."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_csp(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"content-security-policy", _CSP_VALUE.encode()))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_csp)


app.add_middleware(CSPMiddleware)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- ADMIN AUTH ---
security = HTTPBasic()
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin")
ALLOW_INSECURE_ADMIN = os.environ.get("ALLOW_INSECURE_ADMIN", "").lower() in {"1", "true", "yes"}
WEAK_ADMIN_PASSWORDS = {"admin", "password", "change-me", "changeme", "replace-me"}
INSECURE_ADMIN_CREDENTIALS = ADMIN_PASSWORD.lower() in WEAK_ADMIN_PASSWORDS
ADMIN_WS_TOKEN_TTL = int(os.environ.get("ADMIN_WS_TOKEN_TTL", "3600"))
ADMIN_WS_TOKEN_MAX = max(1, int(os.environ.get("ADMIN_WS_TOKEN_MAX", "1000")))
_admin_ws_tokens: dict[str, float] = {}
_admin_ws_tokens_lock = threading.Lock()

if INSECURE_ADMIN_CREDENTIALS:
    if ALLOW_INSECURE_ADMIN:
        logger.warning("Using insecure admin credentials because ALLOW_INSECURE_ADMIN is enabled")
    else:
        logger.critical("Insecure admin credentials are disabled. Set ADMIN_USERNAME and a strong ADMIN_PASSWORD.")


def verify_admin(credentials: HTTPBasicCredentials = Depends(security)):
    if INSECURE_ADMIN_CREDENTIALS and not ALLOW_INSECURE_ADMIN:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Insecure admin credentials are disabled. Set ADMIN_USERNAME and a strong ADMIN_PASSWORD.",
        )
    correct_username = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


def _create_admin_ws_token() -> str:
    token = secrets.token_urlsafe(32)
    expires_at = time.time() + ADMIN_WS_TOKEN_TTL
    with _admin_ws_tokens_lock:
        now = time.time()
        expired = [t for t, exp in _admin_ws_tokens.items() if exp < now]
        for t in expired:
            _admin_ws_tokens.pop(t, None)
        _admin_ws_tokens[token] = expires_at
        while len(_admin_ws_tokens) > ADMIN_WS_TOKEN_MAX:
            _admin_ws_tokens.pop(next(iter(_admin_ws_tokens)))
    return token


def _is_valid_admin_ws_token(token: str) -> bool:
    if not token:
        return False
    with _admin_ws_tokens_lock:
        expires_at = _admin_ws_tokens.get(token)
        if expires_at is None:
            return False
        if expires_at < time.time():
            _admin_ws_tokens.pop(token, None)
            return False
        return True


@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request, credentials: HTTPBasicCredentials = Depends(security)):
    if INSECURE_ADMIN_CREDENTIALS and not ALLOW_INSECURE_ADMIN:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Insecure admin credentials are disabled. Set ADMIN_USERNAME and a strong ADMIN_PASSWORD.",
        )
    # Verify credentials (reuse same logic as verify_admin)
    if not (secrets.compare_digest(credentials.username, ADMIN_USERNAME) and
            secrets.compare_digest(credentials.password, ADMIN_PASSWORD)):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    # Generate short-lived bearer token for browser WebSocket connections.
    ws_token = _create_admin_ws_token()
    return templates.TemplateResponse("admin.html", {"request": request, "ws_token": ws_token})


# --- API ENDPOINTS ---

@app.get("/api/status")
async def api_status(_user: str = Depends(verify_admin), _rl=Depends(rate_limit)):
    cfg = get_config_snapshot()
    with state.audio_stream_lock:
        if state.audio_stream is not None and state.audio_stream.active:
            dev_name = state.cached_device_name or "unknown"
            audio_status = {
                "status": "running",
                "device_name": dev_name,
                "channel": cfg["audio_channel"],
                "native_sample_rate": state.native_sample_rate,
                "resampling_active": state.resampler is not None,
            }
        else:
            audio_status = {"status": "stopped", "device_name": None, "channel": 0}

    device_type = state.device.type if state.device else "unknown"
    gpu_name = None
    if device_type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)

    with state._audio_level_lock:
        current_audio_level = state._audio_level_db

    with state.inference_pending_lock:
        pending = state.inference_pending
    with state._perf_lock:
        perf_snapshot = dict(state._perf_metrics)
    queue_stats = state.get_queue_stats()
    ws_stats = state.manager.get_stats()

    return JSONResponse({
        "status": "ok",
        "clients": state.manager.client_count(),
        "audio_level_db": round(current_audio_level, 1),
        "active_languages": sorted(state.manager.get_unique_languages()),
        "device": device_type,
        "uptime": int(time.time() - state.start_time),
        "components": {
            "model": {"status": "running", "name": MODEL_NAME, "device": device_type, "gpu_name": gpu_name},
            "vad": {"status": "running", "type": "silero"},
            "audio_stream": audio_status,
            "inference_executor": {
                "status": "stuck" if state.inference_stuck.is_set() else "running",
                "pending_tasks": pending,
            },
        },
        "translation_paused": state.translation_paused.is_set(),
        "config": cfg,
        "queues": queue_stats,
        "drops": {
            "audio_queue_drops": perf_snapshot.get("audio_queue_drops", 0),
            "audio_status_drops": perf_snapshot.get("audio_status_drops", 0),
            "audio_buffer_overflows": perf_snapshot.get("audio_buffer_overflows", 0),
            "translation_queue_drops": perf_snapshot.get("translation_queue_drops", 0),
            **ws_stats,
        },
    })


@app.get("/api/devices")
async def api_devices(_user: str = Depends(verify_admin), _rl=Depends(rate_limit)):
    try:
        devices = get_audio_devices()
        return JSONResponse({
            "devices": devices,
            "current_device_index": get_config_snapshot()["audio_device_index"],
            "current_channel": get_config_snapshot()["audio_channel"],
        })
    except Exception as e:
        logger.error(f"Error listing devices: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/devices/select")
async def api_devices_select(request: Request, _user: str = Depends(verify_admin), _rl=Depends(rate_limit)):
    try:
        body = await _read_json_body(request)
        device_index = body.get("device_index")
        channel = body.get("channel", 0)

        if device_index is not None and (not isinstance(device_index, int) or isinstance(device_index, bool)):
            return JSONResponse({"error": "device_index must be int or null"}, status_code=400)
        if not isinstance(channel, int) or isinstance(channel, bool):
            return JSONResponse({"error": "channel must be int"}, status_code=400)

        if device_index is not None:
            devices = sd.query_devices()
            if device_index < 0 or device_index >= len(devices):
                return JSONResponse({"error": f"Invalid device_index: {device_index}"}, status_code=400)
            dev = devices[device_index]
            if dev['max_input_channels'] <= 0:
                return JSONResponse({"error": f"Device {device_index} has no input channels"}, status_code=400)
            if channel < 0 or channel >= dev['max_input_channels']:
                return JSONResponse({"error": f"Channel {channel} out of range (0-{dev['max_input_channels']-1})"}, status_code=400)
        else:
            dev = sd.query_devices(kind='input')
            if dev['max_input_channels'] <= 0:
                return JSONResponse({"error": "Default input device has no input channels"}, status_code=400)
            if channel < 0 or channel >= dev['max_input_channels']:
                return JSONResponse({"error": f"Channel {channel} out of range (0-{dev['max_input_channels']-1})"}, status_code=400)

        await asyncio.to_thread(restart_audio_stream, device_index, channel, state)
        cfg = apply_runtime_config_updates({"audio_device_index": device_index, "audio_channel": channel})
        save_config(cfg)
        return JSONResponse({"ok": True})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error selecting device: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/config")
async def api_config_get(_user: str = Depends(verify_admin), _rl=Depends(rate_limit)):
    return JSONResponse(get_config_snapshot())


def _validate_config_values(body: dict) -> tuple[dict, dict]:
    """Validate config key/value pairs. Returns (validated, errors) dicts."""
    validated = {}
    errors = {}

    if not isinstance(body, dict):
        return {}, {"body": "Expected JSON object"}

    for key, value in body.items():
        if key not in DEFAULT_CONFIG:
            errors[key] = f"Unknown config key: {key}"
            continue

        if key in FIXED_CONFIG_VALUES and value != FIXED_CONFIG_VALUES[key]:
            errors[key] = f"{key} is fixed at {FIXED_CONFIG_VALUES[key]}"
            continue

        expected_type = type(DEFAULT_CONFIG[key])
        if DEFAULT_CONFIG[key] is None:
            if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
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
            if key == "default_target_lang" and value not in SUPPORTED_LANGUAGES:
                errors[key] = f"Unsupported language for {key}: {value}"
                continue

        if key in CONFIG_RANGES:
            min_val, max_val = CONFIG_RANGES[key]
            if value < min_val or value > max_val:
                errors[key] = f"{key} must be between {min_val} and {max_val}"
                continue

        validated[key] = value

    if not errors:
        candidate = get_config_snapshot()
        candidate.update(validated)
        if candidate["min_chunk_duration"] > candidate["max_chunk_duration"]:
            errors["min_chunk_duration"] = "min_chunk_duration must be <= max_chunk_duration"
        if candidate["context_overlap"] >= candidate["min_chunk_duration"]:
            errors["context_overlap"] = "context_overlap must be smaller than min_chunk_duration"

    return validated, errors


@app.post("/api/config")
async def api_config_post(request: Request, _user: str = Depends(verify_admin), _rl=Depends(rate_limit)):
    try:
        body = await _read_json_body(request)
        validated, errors = _validate_config_values(body)

        if errors:
            return JSONResponse({"errors": errors, "config": get_config_snapshot()}, status_code=400)

        old_config = get_config_snapshot()
        candidate_config = {**old_config, **validated}
        if (
            candidate_config["audio_device_index"] != old_config["audio_device_index"]
            or candidate_config["audio_channel"] != old_config["audio_channel"]
        ):
            await asyncio.to_thread(
                restart_audio_stream,
                candidate_config["audio_device_index"],
                candidate_config["audio_channel"],
                state,
            )

        new_config = apply_runtime_config_updates(validated)
        save_config(new_config)
        logger.info(f"Config updated: {body}")

        return JSONResponse(new_config)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/config/export")
async def api_config_export(_user: str = Depends(verify_admin), _rl=Depends(rate_limit)):
    return JSONResponse(get_config_snapshot())


@app.post("/api/config/import")
async def api_config_import(request: Request, _user: str = Depends(verify_admin), _rl=Depends(rate_limit)):
    try:
        body = await _read_json_body(request)
        validated, errors = _validate_config_values(body)

        if errors:
            return JSONResponse({"ok": False, "errors": errors, "imported": 0, "config": get_config_snapshot()}, status_code=400)

        old_config = get_config_snapshot()
        candidate_config = {**old_config, **validated}
        if (
            candidate_config["audio_device_index"] != old_config["audio_device_index"]
            or candidate_config["audio_channel"] != old_config["audio_channel"]
        ):
            await asyncio.to_thread(
                restart_audio_stream,
                candidate_config["audio_device_index"],
                candidate_config["audio_channel"],
                state,
            )

        new_config = apply_runtime_config_updates(validated)
        if validated:
            save_config(new_config)
        logger.info(f"Config imported: {len(validated)} keys" + (f", {len(errors)} rejected" if errors else ""))
        result = {"ok": True, "imported": len(validated), "config": new_config}
        if errors:
            result["errors"] = errors
        return JSONResponse(result)
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


def _health_payload() -> dict:
    with state.audio_stream_lock:
        audio_ok = state.audio_stream is not None and state.audio_stream.active
    model_ok = state.processor is not None and state.model is not None
    thread_ok = state.processing_thread is not None and state.processing_thread.is_alive()
    broadcaster_ok = state.broadcaster_task is not None and not state.broadcaster_task.done()
    inference_ok = not state.inference_stuck.is_set()

    gpu_ok = True
    if state.device and state.device.type == "cuda":
        try:
            torch.cuda.mem_get_info(0)
        except Exception:
            gpu_ok = False

    if model_ok and thread_ok and broadcaster_ok and audio_ok and gpu_ok and inference_ok:
        health_status = "healthy"
    elif model_ok and thread_ok and broadcaster_ok:
        health_status = "degraded"
    else:
        health_status = "unhealthy"

    return {
        "status": health_status,
        "uptime": int(time.time() - state.start_time),
        "clients": state.manager.client_count(),
        "audio_stream": "running" if audio_ok else "stopped",
        "model": "loaded" if model_ok else "not_loaded",
        "processing_thread": "running" if thread_ok else "stopped",
        "broadcaster": "running" if broadcaster_ok else "stopped",
        "inference": "stuck" if state.inference_stuck.is_set() else "running",
        "gpu": "ok" if gpu_ok else "error",
        "admin_auth": "insecure_disabled" if INSECURE_ADMIN_CREDENTIALS and not ALLOW_INSECURE_ADMIN else "configured",
        "queues": state.get_queue_stats(),
    }


@app.get("/health")
async def health_check():
    return JSONResponse(_health_payload())


@app.get("/ready")
async def readiness_check():
    payload = _health_payload()
    return JSONResponse(payload, status_code=200 if payload["status"] == "healthy" else 503)


_qr_cache = {"url": None, "svg": None}
_qr_cache_lock = threading.Lock()


@app.get("/api/qr.svg")
async def qr_code_svg(request: Request):
    """Generate QR code SVG dynamically from the request URL (cached)."""
    import qrcode
    import qrcode.image.svg
    import io

    base_url = str(request.base_url).rstrip("/")
    with _qr_cache_lock:
        if _qr_cache["url"] == base_url and _qr_cache["svg"] is not None:
            return Response(content=_qr_cache["svg"], media_type="image/svg+xml",
                            headers={"Cache-Control": "no-cache"})

    img = qrcode.make(base_url, image_factory=qrcode.image.svg.SvgPathImage)
    buf = io.BytesIO()
    img.save(buf)
    svg_bytes = buf.getvalue()
    with _qr_cache_lock:
        _qr_cache["url"] = base_url
        _qr_cache["svg"] = svg_bytes
    return Response(content=svg_bytes, media_type="image/svg+xml",
                    headers={"Cache-Control": "no-cache"})


@app.get("/api/sessions")
async def api_sessions(_user: str = Depends(verify_admin), _rl=Depends(rate_limit)):
    return JSONResponse(state.manager.get_sessions_info())


@app.get("/api/metrics")
async def api_metrics(_user: str = Depends(verify_admin), _rl=Depends(rate_limit)):
    with state._perf_lock:
        enc = list(state._perf_metrics["encoder_ms"])
        dec = list(state._perf_metrics["decoder_ms"])
        total = state._perf_metrics["total_translations"]
        last_time = state._perf_metrics["last_inference_time"]
        drops = {
            "audio_queue_drops": state._perf_metrics.get("audio_queue_drops", 0),
            "audio_status_drops": state._perf_metrics.get("audio_status_drops", 0),
            "audio_buffer_overflows": state._perf_metrics.get("audio_buffer_overflows", 0),
            "translation_queue_drops": state._perf_metrics.get("translation_queue_drops", 0),
            "stuck_inference_count": state._perf_metrics.get("stuck_inference_count", 0),
            "processing_loop_errors": state._perf_metrics.get("processing_loop_errors", 0),
        }
    queue_stats = state.get_queue_stats()
    ws_stats = state.manager.get_stats()
    device_type = state.device.type if state.device else "unknown"
    return JSONResponse({
        "total_translations": total,
        "avg_encoder_ms": round(sum(enc) / len(enc), 1) if enc else 0,
        "avg_decoder_ms": round(sum(dec) / len(dec), 1) if dec else 0,
        "last_encoder_ms": round(enc[-1], 1) if enc else 0,
        "last_decoder_ms": round(dec[-1], 1) if dec else 0,
        "last_inference_ago": round(time.time() - last_time, 1) if last_time else None,
        "gpu_name": torch.cuda.get_device_name(0) if device_type == "cuda" else None,
        "gpu_memory_used_mb": round(torch.cuda.memory_allocated(0) / 1024 / 1024, 0) if device_type == "cuda" else None,
        "gpu_memory_total_mb": round(torch.cuda.get_device_properties(0).total_mem / 1024 / 1024, 0) if device_type == "cuda" else None,
        "inference_stuck": state.inference_stuck.is_set(),
        "inference_stuck_for": round(time.time() - state.inference_stuck_since, 1) if state.inference_stuck.is_set() else 0,
        **queue_stats,
        **drops,
        **ws_stats,
    })


@app.get("/api/translations")
async def api_translations(_user: str = Depends(verify_admin), _rl=Depends(rate_limit)):
    with state._translation_history_lock:
        return JSONResponse(list(state._translation_history))


@app.post("/api/translation/toggle")
async def api_translation_toggle(_user: str = Depends(verify_admin), _rl=Depends(rate_limit)):
    if state.translation_paused.is_set():
        state.translation_paused.clear()
        paused = False
    else:
        state.translation_paused.set()
        paused = True
    return JSONResponse({"ok": True, "paused": paused})


@app.get("/api/translation/status")
async def api_translation_status(_user: str = Depends(verify_admin), _rl=Depends(rate_limit)):
    return JSONResponse({"paused": state.translation_paused.is_set()})


@app.get("/api/audio-history")
async def api_audio_history(_user: str = Depends(verify_admin), _rl=Depends(rate_limit)):
    with state._audio_level_lock:
        return JSONResponse(list(state._audio_history))


# --- WebSocket endpoints ---
WS_SEND_TIMEOUT = float(os.environ.get("WS_SEND_TIMEOUT", "2.0"))

def _verify_ws_auth(websocket: WebSocket) -> bool:
    """Verify admin credentials for WebSocket connections.

    Checks Authorization header first (works for non-browser clients),
    then falls back to ?token= query parameter (Base64-encoded user:pass)
    because browser WebSocket API does not support custom headers.
    """
    if INSECURE_ADMIN_CREDENTIALS and not ALLOW_INSECURE_ADMIN:
        return False
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
    # Fallback: ?token=<short-lived bearer token> for browser WebSocket
    token = websocket.query_params.get("token", "")
    if _is_valid_admin_ws_token(token):
        return True
    return False


@app.websocket("/api/status/ws")
async def api_status_ws(websocket: WebSocket):
    if not _verify_ws_auth(websocket):
        await websocket.close(code=4401)
        return
    await websocket.accept()
    try:
        while True:
            cfg = get_config_snapshot()
            with state.audio_stream_lock:
                if state.audio_stream is not None and state.audio_stream.active:
                    dev_name = state.cached_device_name or "unknown"
                    audio_status = {"status": "running", "device_name": dev_name, "channel": cfg["audio_channel"]}
                else:
                    audio_status = {"status": "stopped", "device_name": None, "channel": 0}
            with state._audio_level_lock:
                current_db = state._audio_level_db
                current_peak = state._audio_level_peak
            with state.inference_pending_lock:
                pending = state.inference_pending
            device_type = state.device.type if state.device else "unknown"
            gpu_name = torch.cuda.get_device_name(0) if device_type == "cuda" else None
            payload = {
                "clients": state.manager.client_count(),
                "audio_level_db": round(current_db, 1),
                "audio_level_peak": round(current_peak, 1),
                "active_languages": sorted(state.manager.get_unique_languages()),
                "device": device_type,
                "uptime": int(time.time() - state.start_time),
                "components": {
                    "model": {"status": "running", "name": MODEL_NAME, "device": device_type, "gpu_name": gpu_name},
                    "vad": {"status": "running", "type": "silero"},
                    "audio_stream": audio_status,
                    "inference_executor": {"status": "stuck" if state.inference_stuck.is_set() else "running", "pending_tasks": pending},
                },
                "translation_paused": state.translation_paused.is_set(),
                "queues": state.get_queue_stats(),
            }
            await asyncio.wait_for(websocket.send_text(json.dumps(payload)), timeout=WS_SEND_TIMEOUT)
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug(f"Status websocket closed: {e}")


@app.websocket("/api/logs")
async def api_logs_ws(websocket: WebSocket):
    if not _verify_ws_auth(websocket):
        await websocket.close(code=4401)
        return
    await websocket.accept()
    listener = log_handler.add_listener()
    try:
        history = log_handler.get_history()
        await asyncio.wait_for(
            websocket.send_text(json.dumps({"type": "history", "entries": history})),
            timeout=WS_SEND_TIMEOUT,
        )

        while True:
            try:
                entry = await asyncio.wait_for(listener.get(), timeout=15.0)
                payload = {"type": "log", "entry": entry}
            except asyncio.TimeoutError:
                payload = {"type": "ping"}
            await asyncio.wait_for(websocket.send_text(json.dumps(payload)), timeout=WS_SEND_TIMEOUT)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug(f"Log websocket closed: {e}")
    finally:
        log_handler.remove_listener(listener)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    session_id, reject_reason = await state.manager.connect_with_reason(websocket)
    if session_id is None:
        await websocket.close(code=1013, reason=reject_reason)
        logger.warning(f"Client rejected: {reject_reason}")
        return
    logger.info(f"Client connected: {session_id[:8]}")
    try:
        while True:
            msg = await websocket.receive_text()
            if len(msg) > 1024:
                await websocket.close(1009, "Message too large")
                return
            try:
                data = json.loads(msg)
            except (json.JSONDecodeError, ValueError):
                logger.warning(f"Session {session_id[:8]} sent invalid JSON")
                continue
            msg_type = data.get("type")
            if msg_type == "ping":
                await state.manager.send_to_session(session_id, '{"type":"pong"}')
            elif msg_type == "set_lang" and "lang" in data:
                lang = data["lang"]
                if not isinstance(lang, str) or lang not in SUPPORTED_LANGUAGES:
                    logger.warning(f"Session {session_id[:8]} invalid language: {str(lang)[:20]}")
                    continue
                state.manager.set_language(session_id, lang)
                logger.info(f"Session {session_id[:8]} language -> {lang}")
    except WebSocketDisconnect:
        pass
    finally:
        state.manager.disconnect(session_id)
        logger.info(f"Client disconnected: {session_id[:8]}")


# --- Processing loop ---

def submit_translation(audio_np, target_langs, config_snapshot, loop):
    """Run translation in ThreadPoolExecutor and push results to translation_queue."""
    try:
        results = translate_audio(
            audio_np, sorted(target_langs),
            state.processor, state.model, state.device, state.dtype,
            config_snapshot,
            state._perf_metrics, state._perf_lock,
            state._translation_history, state._translation_history_lock,
        )
        if results:
            def _enqueue_results():
                try:
                    state.translation_queue.put_nowait(results)
                except asyncio.QueueFull:
                    state.increment_metric("translation_queue_drops")
                    logger.warning("Translation queue full; dropping result")

            try:
                loop.call_soon_threadsafe(_enqueue_results)
            except RuntimeError as e:
                state.increment_metric("translation_queue_drops")
                logger.warning(f"Event loop unavailable; dropping translation result: {e}")
    except Exception as e:
        logger.error(f"Translation Error: {e}")
    finally:
        with state.inference_pending_lock:
            state.inference_pending = max(0, state.inference_pending - 1)
        state.inference_stuck.clear()
        state.inference_stuck_since = 0.0


INFERENCE_TIMEOUT = 30  # seconds — log critical if inference exceeds this
SHUTDOWN_TRANSLATION_FLUSH = os.environ.get("SHUTDOWN_TRANSLATION_FLUSH", "").lower() in {"1", "true", "yes"}


def processing_loop(loop):
    # Buffer stores audio at NATIVE sample rate (no per-chunk resampling)
    native_sr = state.native_sample_rate
    config_snapshot = get_config_snapshot()
    current_max_chunk = config_snapshot["max_chunk_duration"]
    MAX_BUFFER_SAMPLES = native_sr * int(current_max_chunk + 5)
    audio_buffer = np.zeros(MAX_BUFFER_SAMPLES, dtype=np.float32)
    buffer_pos = 0
    prev_audio = np.array([], dtype=np.float32)  # stored at native SR
    silence_start = None
    is_speaking = False
    VAD_MIN_SAMPLES = 512
    last_future = None
    last_future_time = 0
    last_vad_reset = time.time()
    VAD_RESET_INTERVAL = 300  # reset VAD states every 5 minutes to prevent drift

    # Pre-allocated VAD buffer at 16kHz (VAD requires 16kHz)
    VAD_BUFFER_MAX = 16000  # ~1s at 16kHz
    vad_np_buffer = np.zeros(VAD_BUFFER_MAX, dtype=np.float32)
    vad_pos = 0
    vad_buffer_time = time.time()
    vad_chunk_tensor = torch.zeros(VAD_MIN_SAMPLES, dtype=torch.float32)

    _level_update_interval = 0.1
    _last_level_update = 0.0
    _pending_level = -60.0
    _pending_peak = -60.0
    _last_overflow_warn = 0.0

    while not state.stop_event.is_set():
        try:
            config_snapshot = get_config_snapshot()
            chunk = state.audio_queue.get(timeout=0.1)
            chunk_np = np.asarray(chunk, dtype=np.float32).reshape(-1)

            # Periodic VAD state reset to prevent RNN drift
            now_vad_reset = time.time()
            if now_vad_reset - last_vad_reset > VAD_RESET_INTERVAL:
                state.vad_model.reset_states()
                last_vad_reset = now_vad_reset
                logger.debug("VAD states reset (periodic)")

            # Detect SR change (device switch) — reset buffer
            current_native_sr = state.native_sample_rate
            if current_native_sr != native_sr:
                logger.info(f"Sample rate changed {native_sr} -> {current_native_sr}, resetting buffer")
                native_sr = current_native_sr
                current_max_chunk = config_snapshot["max_chunk_duration"]
                MAX_BUFFER_SAMPLES = native_sr * int(current_max_chunk + 5)
                audio_buffer = np.zeros(MAX_BUFFER_SAMPLES, dtype=np.float32)
                buffer_pos = 0
                prev_audio = np.array([], dtype=np.float32)
                is_speaking = False
                silence_start = None
                vad_pos = 0
                state.vad_model.reset_states()
                last_vad_reset = now_vad_reset

            # Detect max_chunk_duration change — resize buffer
            new_max_chunk = config_snapshot["max_chunk_duration"]
            if new_max_chunk != current_max_chunk:
                logger.info(f"max_chunk_duration changed {current_max_chunk} -> {new_max_chunk}, resizing buffer")
                current_max_chunk = new_max_chunk
                new_max_samples = native_sr * int(new_max_chunk + 5)
                new_buffer = np.zeros(new_max_samples, dtype=np.float32)
                keep = min(buffer_pos, new_max_samples)
                if keep > 0:
                    new_buffer[:keep] = audio_buffer[buffer_pos - keep:buffer_pos]
                audio_buffer = new_buffer
                buffer_pos = keep
                MAX_BUFFER_SAMPLES = new_max_samples

            # Grab resampler for VAD (chunk_np stays at native SR for main buffer)
            with state.audio_stream_lock:
                current_resampler = state.resampler

            # Resample only for VAD (16kHz required by Silero)
            chunk_16k = resample_audio(chunk_np, current_resampler)

            # Audio level computed on native SR (RMS/dB is SR-independent)
            level = compute_audio_level(chunk_np)
            if level > _pending_level:
                _pending_level = level
            if level > _pending_peak:
                _pending_peak = level

            now_lvl = time.time()
            if now_lvl - _last_level_update >= _level_update_interval:
                with state._audio_level_lock:
                    state._audio_level_db = _pending_level
                    if _pending_peak > state._audio_level_peak:
                        state._audio_level_peak = _pending_peak
                _pending_level = -60.0
                _pending_peak = -60.0
                _last_level_update = now_lvl

            # Store chunk at native SR in main buffer
            n = chunk_np.shape[0]
            if buffer_pos + n <= MAX_BUFFER_SAMPLES:
                audio_buffer[buffer_pos:buffer_pos + n] = chunk_np
                buffer_pos += n
            else:
                state.increment_metric("audio_buffer_overflows")
                now_overflow = time.time()
                if now_overflow - _last_overflow_warn > 5.0:
                    logger.warning(f"Audio buffer overflow: dropping {n} samples (buffer full at {buffer_pos}/{MAX_BUFFER_SAMPLES})")
                    _last_overflow_warn = now_overflow

            # VAD buffer uses 16kHz resampled data
            vad_n = min(chunk_16k.shape[0], VAD_BUFFER_MAX - vad_pos)
            if vad_n > 0:
                vad_np_buffer[vad_pos:vad_pos + vad_n] = chunk_16k[:vad_n]
                vad_pos += vad_n
            vad_buffer_time = time.time()
            speech_detected = is_speaking

            # Process VAD in 512-sample steps
            vad_read_pos = 0
            while vad_pos - vad_read_pos >= VAD_MIN_SAMPLES:
                vad_chunk_tensor.copy_(torch.from_numpy(vad_np_buffer[vad_read_pos:vad_read_pos + VAD_MIN_SAMPLES]))
                vad_read_pos += VAD_MIN_SAMPLES
                with torch.inference_mode():
                    confidence = state.vad_model(vad_chunk_tensor, 16000).item()
                speech_detected = confidence > 0.5
            remaining = vad_pos - vad_read_pos
            if remaining > 0 and vad_read_pos > 0:
                vad_np_buffer[:remaining] = vad_np_buffer[vad_read_pos:vad_read_pos + remaining]
            vad_pos = remaining

            # Duration calculations use native SR
            total_duration = buffer_pos / native_sr
            current_time = time.time()

            if speech_detected:
                is_speaking = True
                silence_start = None
            else:
                if silence_start is None:
                    silence_start = current_time

            silence_dur = config_snapshot["silence_duration"]
            max_chunk = config_snapshot["max_chunk_duration"]
            min_chunk = config_snapshot["min_chunk_duration"]
            ctx_overlap = config_snapshot["context_overlap"]

            should_process = False
            if is_speaking and silence_start and (current_time - silence_start > silence_dur):
                should_process = True
            if total_duration > max_chunk:
                should_process = True

            if should_process and total_duration >= min_chunk:
                # If translation is paused, discard the chunk
                if state.translation_paused.is_set():
                    overlap_samples = int(ctx_overlap * native_sr)
                    if overlap_samples > 0 and buffer_pos > overlap_samples:
                        prev_audio = audio_buffer[buffer_pos - overlap_samples:buffer_pos].copy()
                    else:
                        prev_audio = np.array([], dtype=np.float32)
                    buffer_pos = 0
                    is_speaking = False
                    silence_start = None
                    continue

                full_audio = audio_buffer[:buffer_pos].copy()
                target_langs = state.manager.get_unique_languages()
                if target_langs:
                    audio_to_process = full_audio
                    if prev_audio.size > 0:
                        audio_to_process = np.concatenate((prev_audio, full_audio))

                    # Resample entire segment ONCE before sending to model
                    audio_to_process = resample_audio(audio_to_process, current_resampler)

                    # Check for stuck inference
                    if last_future is not None and not last_future.done():
                        elapsed = time.time() - last_future_time
                        if elapsed > INFERENCE_TIMEOUT:
                            now = time.time()
                            if not state.inference_stuck.is_set():
                                state.inference_stuck.set()
                                state.inference_stuck_since = now
                                state._last_stuck_log = now
                                with state._perf_lock:
                                    state._perf_metrics["stuck_inference_count"] += 1
                                logger.critical(
                                    f"Inference exceeded {INFERENCE_TIMEOUT}s ({elapsed:.0f}s). "
                                    "Holding new work until the current inference returns; restart the process if it does not recover."
                                )
                            elif now - state._last_stuck_log > 30.0:
                                logger.critical(f"Inference still stuck for {elapsed:.0f}s; dropping new audio segment")
                                state._last_stuck_log = now
                        else:
                            logger.warning(f"Skipping chunk ({total_duration:.1f}s) — inference busy ({elapsed:.0f}s)")
                    else:
                        if state.inference_stuck.is_set():
                            logger.warning("Inference recovered; accepting new work")
                            state.inference_stuck.clear()
                            state.inference_stuck_since = 0.0
                        with state.inference_pending_lock:
                            if state.inference_pending >= 1:
                                logger.warning(f"Skipping chunk ({total_duration:.1f}s) — inference queue busy")
                            else:
                                state.inference_pending += 1
                                last_future = state.inference_executor.submit(
                                    submit_translation, audio_to_process, target_langs, config_snapshot, loop
                                )
                                last_future_time = time.time()

                # Context overlap stored at native SR
                samples_ctx = int(native_sr * ctx_overlap)
                if full_audio.shape[0] > samples_ctx:
                    prev_audio = full_audio[-samples_ctx:]
                else:
                    prev_audio = full_audio

                buffer_pos = 0
                is_speaking = False
                silence_start = None

                # Reset VAD after each speech segment to prevent state drift
                state.vad_model.reset_states()
                last_vad_reset = time.time()

            elif not is_speaking and total_duration > min_chunk:
                samples_to_keep = int(native_sr * 1.0)
                if buffer_pos > samples_to_keep * 5:
                    buffer_pos = 0

        except queue.Empty:
            if vad_pos > 0 and time.time() - vad_buffer_time > 5.0:
                vad_pos = 0
            continue
        except Exception as e:
            state.increment_metric("processing_loop_errors")
            logger.error(f"Loop Error: {e}")

    # Flush remaining buffer on shutdown
    if not SHUTDOWN_TRANSLATION_FLUSH:
        if buffer_pos > 0:
            logger.info("Dropping buffered audio on shutdown; set SHUTDOWN_TRANSLATION_FLUSH=1 to translate final buffered segment")
        return

    config_snapshot = get_config_snapshot()
    min_chunk = config_snapshot["min_chunk_duration"]
    if buffer_pos > 0 and buffer_pos / native_sr >= min_chunk:
        full_audio = audio_buffer[:buffer_pos].copy().astype(np.float32)
        if prev_audio.size > 0:
            full_audio = np.concatenate((prev_audio, full_audio))
        # Resample before sending to model
        with state.audio_stream_lock:
            current_resampler = state.resampler
        full_audio = resample_audio(full_audio, current_resampler)
        target_langs = state.manager.get_unique_languages()
        if target_langs:
            try:
                with state.inference_pending_lock:
                    state.inference_pending += 1
                submit_translation(full_audio, target_langs, config_snapshot, loop)
            except Exception as e:
                logger.error(f"Final flush translation error: {e}")


async def broadcaster():
    """Send translated texts to clients per their language."""
    while not state.stop_event.is_set():
        try:
            results = await asyncio.wait_for(state.translation_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            continue
        try:
            tasks = []
            for lang, text in results.items():
                logger.debug(f"Broadcasting translation for {lang} ({len(text)} chars)")
                tasks.append(state.manager.send_to_language(lang, text))
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=False)
        except Exception as e:
            logger.error(f"Broadcast error: {e}")
        finally:
            state.translation_queue.task_done()


async def audio_history_ticker():
    """Record audio level once per second for history graph."""
    while not state.stop_event.is_set():
        await asyncio.sleep(1)
        state._update_audio_history()


def audio_watchdog():
    """Monitor audio device health and attempt reconnection on failure.

    Runs in its own thread. Uses state.audio_device_error (threading.Event)
    signaled by audio callbacks when device errors occur.
    """
    RECONNECT_DELAYS = [1, 2, 4, 8, 16, 30]  # exponential backoff, max 30s

    while not state.stop_event.is_set():
        # Wait for error signal or periodic check (5s)
        signaled = state.audio_device_error.wait(timeout=5.0)

        if state.stop_event.is_set():
            break

        # Check if stream is dead (either error signaled or stream inactive)
        needs_restart = signaled
        if not needs_restart:
            with state.audio_stream_lock:
                if state.audio_stream is not None and not state.audio_stream.active:
                    needs_restart = True

        if not needs_restart:
            continue

        state.audio_device_error.clear()
        logger.warning("Audio device error detected, attempting reconnection...")

        for attempt, delay in enumerate(RECONNECT_DELAYS):
            if state.stop_event.is_set():
                return
            try:
                cfg = get_config_snapshot()
                device_idx = cfg["audio_device_index"]
                restart_audio_stream(device_idx, cfg["audio_channel"], state)
                logger.info(f"Audio device reconnected (attempt {attempt + 1})")
                break
            except Exception as e:
                logger.warning(f"Audio reconnect attempt {attempt + 1} failed: {e}")
                # Wait before next attempt, but check stop_event
                for _ in range(int(delay * 10)):
                    if state.stop_event.is_set():
                        return
                    time.sleep(0.1)
        else:
            logger.error("Audio reconnection failed after all attempts. Waiting for next error signal.")


def start_server():
    """Start the uvicorn server."""
    import uvicorn

    cfg = get_config_snapshot()
    device_idx = cfg["audio_device_index"]
    try:
        restart_audio_stream(device_idx, cfg["audio_channel"], state)
    except Exception as e:
        logger.error(f"Initial audio stream failed: {e} — will retry via watchdog")

    @app.on_event("startup")
    async def startup_event():
        loop = asyncio.get_running_loop()
        state.broadcaster_task = asyncio.create_task(broadcaster())
        state.audio_ticker_task = asyncio.create_task(audio_history_ticker())
        # Audio watchdog runs in its own thread (uses blocking wait/sleep)
        state.watchdog_thread = threading.Thread(target=audio_watchdog, daemon=False)
        state.watchdog_thread.start()
        # Processing thread — NOT daemon, coordinated shutdown via stop_event
        state.processing_thread = threading.Thread(target=processing_loop, args=(loop,))
        state.processing_thread.start()

    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Shutting down...")
        # 1. Signal all threads to stop
        state.stop_event.set()
        state.audio_device_error.set()  # unblock watchdog

        # 2. Cancel async tasks
        for task_ref in (state.broadcaster_task, state.audio_ticker_task):
            if task_ref and not task_ref.done():
                task_ref.cancel()
                try:
                    await asyncio.wait_for(task_ref, timeout=3)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass

        # 3. Wait for processing thread to finish current work
        if state.processing_thread and state.processing_thread.is_alive():
            state.processing_thread.join(timeout=10)
            if state.processing_thread.is_alive():
                logger.warning("Processing thread did not stop in time")

        # 4. Wait for watchdog thread
        if state.watchdog_thread and state.watchdog_thread.is_alive():
            state.watchdog_thread.join(timeout=5)

        # 5. Shutdown inference executor
        state.inference_executor.shutdown(wait=False)

        # 4. Stop audio stream
        with state.audio_stream_lock:
            if state.audio_stream is not None:
                try:
                    state.audio_stream.stop()
                except Exception:
                    pass
                try:
                    state.audio_stream.close()
                except Exception:
                    pass
                state.audio_stream = None

        # 5. Save config
        save_config(get_config_snapshot())

        # 6. Release GPU memory
        if state.device and state.device.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        logger.info("Shutdown complete.")

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8888"))
    ws_max_size = int(os.environ.get("WS_MAX_SIZE", "2048"))
    timeout_keep_alive = int(os.environ.get("UVICORN_TIMEOUT_KEEP_ALIVE", "5"))
    proxy_headers = os.environ.get("UVICORN_PROXY_HEADERS", "").lower() in {"1", "true", "yes"}
    forwarded_allow_ips = os.environ.get("FORWARDED_ALLOW_IPS", "127.0.0.1")
    logger.info(f"Starting Web Server on {host}:{port}...")
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        ws_max_size=ws_max_size,
        timeout_keep_alive=timeout_keep_alive,
        proxy_headers=proxy_headers,
        forwarded_allow_ips=forwarded_allow_ips,
    )
