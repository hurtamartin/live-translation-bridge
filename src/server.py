import asyncio
import base64
import json
import os
import queue
import threading
import time

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
from src.config import runtime_config, save_config, DEFAULT_CONFIG, CONFIG_RANGES, SUPPORTED_LANGUAGES

# --- Simple in-memory rate limiter ---
_rate_limit_store: dict[str, list[float]] = {}
_rate_limit_lock = threading.Lock()
RATE_LIMIT_MAX = int(os.environ.get("RATE_LIMIT_MAX", "10"))  # requests per window
RATE_LIMIT_WINDOW = 1.0  # seconds


def _check_rate_limit(client_ip: str) -> bool:
    """Return True if request is allowed, False if rate-limited."""
    now = time.time()
    with _rate_limit_lock:
        # Cleanup stale entries when store grows too large
        if len(_rate_limit_store) > 10000:
            cutoff = now - RATE_LIMIT_WINDOW * 2
            stale_keys = [k for k, v in _rate_limit_store.items() if all(t < cutoff for t in v)]
            for k in stale_keys:
                del _rate_limit_store[k]

        timestamps = _rate_limit_store.get(client_ip, [])
        # Remove expired entries
        timestamps = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW]
        if len(timestamps) >= RATE_LIMIT_MAX:
            _rate_limit_store[client_ip] = timestamps
            return False
        timestamps.append(now)
        _rate_limit_store[client_ip] = timestamps
        return True


def rate_limit(request: Request):
    """FastAPI dependency for rate limiting admin API endpoints."""
    client_ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Too many requests")
from src.translation.engine import MODEL_NAME, translate_audio
from src.audio.capture import (
    get_audio_devices, restart_audio_stream, compute_audio_level,
    resample_audio, is_speech,
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


@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request, credentials: HTTPBasicCredentials = Depends(security)):
    # Verify credentials (reuse same logic as verify_admin)
    if not (secrets.compare_digest(credentials.username, ADMIN_USERNAME) and
            secrets.compare_digest(credentials.password, ADMIN_PASSWORD)):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    # Generate token for authenticated WebSocket connections
    ws_token = base64.b64encode(f"{credentials.username}:{credentials.password}".encode()).decode()
    return templates.TemplateResponse("admin.html", {"request": request, "ws_token": ws_token})


# --- API ENDPOINTS ---

@app.get("/api/status")
async def api_status(_user: str = Depends(verify_admin), _rl=Depends(rate_limit)):
    with state.audio_stream_lock:
        if state.audio_stream is not None and state.audio_stream.active:
            dev_name = state.cached_device_name or "unknown"
            audio_status = {
                "status": "running",
                "device_name": dev_name,
                "channel": runtime_config["audio_channel"],
                "native_sample_rate": state.native_sample_rate,
                "resampling_active": state.resampler is not None,
            }
        else:
            audio_status = {"status": "stopped", "device_name": None, "channel": 0}

    gpu_name = None
    if state.device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)

    with state._audio_level_lock:
        current_audio_level = state._audio_level_db

    with state.inference_pending_lock:
        pending = state.inference_pending

    return JSONResponse({
        "status": "ok",
        "clients": state.manager.client_count(),
        "audio_level_db": round(current_audio_level, 1),
        "active_languages": sorted(state.manager.get_unique_languages()),
        "device": state.device.type,
        "uptime": int(time.time() - state.start_time),
        "components": {
            "model": {"status": "running", "name": MODEL_NAME, "device": state.device.type, "gpu_name": gpu_name},
            "vad": {"status": "running", "type": "silero"},
            "audio_stream": audio_status,
            "inference_executor": {
                "status": "running",
                "pending_tasks": pending,
            },
        },
        "config": dict(runtime_config),
    })


@app.get("/api/devices")
async def api_devices(_user: str = Depends(verify_admin), _rl=Depends(rate_limit)):
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
async def api_devices_select(request: Request, _user: str = Depends(verify_admin), _rl=Depends(rate_limit)):
    try:
        body = await request.json()
        device_index = body.get("device_index")
        channel = body.get("channel", 0)

        if device_index is not None:
            devices = sd.query_devices()
            if device_index < 0 or device_index >= len(devices):
                return JSONResponse({"error": f"Invalid device_index: {device_index}"}, status_code=400)
            dev = devices[device_index]
            if dev['max_input_channels'] <= 0:
                return JSONResponse({"error": f"Device {device_index} has no input channels"}, status_code=400)
            if channel < 0 or channel >= dev['max_input_channels']:
                return JSONResponse({"error": f"Channel {channel} out of range (0-{dev['max_input_channels']-1})"}, status_code=400)

        runtime_config["audio_device_index"] = device_index
        runtime_config["audio_channel"] = channel

        restart_audio_stream(device_index, channel, state)
        return JSONResponse({"ok": True})
    except Exception as e:
        logger.error(f"Error selecting device: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/config")
async def api_config_get(_user: str = Depends(verify_admin), _rl=Depends(rate_limit)):
    return JSONResponse(dict(runtime_config))


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


@app.get("/health")
async def health_check():
    with state.audio_stream_lock:
        audio_ok = state.audio_stream is not None and state.audio_stream.active
    model_ok = state.processor is not None and state.model is not None
    thread_ok = state.processing_thread is not None and state.processing_thread.is_alive()

    gpu_ok = True
    if state.device and state.device.type == "cuda":
        try:
            torch.cuda.mem_get_info(0)
        except Exception:
            gpu_ok = False

    if model_ok and thread_ok and audio_ok and gpu_ok:
        status = "healthy"
    elif model_ok and thread_ok:
        status = "degraded"
    else:
        status = "unhealthy"

    return JSONResponse({
        "status": status,
        "uptime": int(time.time() - state.start_time),
        "clients": state.manager.client_count(),
        "audio_stream": "running" if audio_ok else "stopped",
        "model": "loaded" if model_ok else "not_loaded",
        "processing_thread": "running" if thread_ok else "stopped",
        "gpu": "ok" if gpu_ok else "error",
    })


_qr_cache = {"url": None, "svg": None}


@app.get("/api/qr.svg")
async def qr_code_svg(request: Request):
    """Generate QR code SVG dynamically from the request URL (cached)."""
    import qrcode
    import qrcode.image.svg
    import io

    base_url = str(request.base_url).rstrip("/")
    if _qr_cache["url"] == base_url and _qr_cache["svg"] is not None:
        return Response(content=_qr_cache["svg"], media_type="image/svg+xml",
                        headers={"Cache-Control": "no-cache"})

    img = qrcode.make(base_url, image_factory=qrcode.image.svg.SvgPathImage)
    buf = io.BytesIO()
    img.save(buf)
    svg_bytes = buf.getvalue()
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
    return JSONResponse({
        "total_translations": total,
        "avg_encoder_ms": round(sum(enc) / len(enc), 1) if enc else 0,
        "avg_decoder_ms": round(sum(dec) / len(dec), 1) if dec else 0,
        "last_encoder_ms": round(enc[-1], 1) if enc else 0,
        "last_decoder_ms": round(dec[-1], 1) if dec else 0,
        "last_inference_ago": round(time.time() - last_time, 1) if last_time else None,
        "gpu_name": torch.cuda.get_device_name(0) if state.device.type == "cuda" else None,
        "gpu_memory_used_mb": round(torch.cuda.memory_allocated(0) / 1024 / 1024, 0) if state.device.type == "cuda" else None,
        "gpu_memory_total_mb": round(torch.cuda.get_device_properties(0).total_mem / 1024 / 1024, 0) if state.device.type == "cuda" else None,
    })


@app.get("/api/translations")
async def api_translations(_user: str = Depends(verify_admin), _rl=Depends(rate_limit)):
    with state._translation_history_lock:
        return JSONResponse(list(state._translation_history))


@app.get("/api/audio-history")
async def api_audio_history(_user: str = Depends(verify_admin), _rl=Depends(rate_limit)):
    with state._audio_level_lock:
        return JSONResponse(list(state._audio_history))


# --- WebSocket endpoints ---

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


@app.websocket("/api/status/ws")
async def api_status_ws(websocket: WebSocket):
    if not _verify_ws_auth(websocket):
        await websocket.close(code=4401)
        return
    await websocket.accept()
    try:
        while True:
            with state.audio_stream_lock:
                if state.audio_stream is not None and state.audio_stream.active:
                    dev_name = state.cached_device_name or "unknown"
                    audio_status = {"status": "running", "device_name": dev_name, "channel": runtime_config["audio_channel"]}
                else:
                    audio_status = {"status": "stopped", "device_name": None, "channel": 0}
            with state._audio_level_lock:
                current_db = state._audio_level_db
                current_peak = state._audio_level_peak
                state._audio_level_peak = max(state._audio_level_peak - 0.5, state._audio_level_db)
            with state.inference_pending_lock:
                pending = state.inference_pending
            gpu_name = torch.cuda.get_device_name(0) if state.device.type == "cuda" else None
            payload = {
                "clients": state.manager.client_count(),
                "audio_level_db": round(current_db, 1),
                "audio_level_peak": round(current_peak, 1),
                "active_languages": sorted(state.manager.get_unique_languages()),
                "device": state.device.type,
                "uptime": int(time.time() - state.start_time),
                "components": {
                    "model": {"status": "running", "name": MODEL_NAME, "device": state.device.type, "gpu_name": gpu_name},
                    "vad": {"status": "running", "type": "silero"},
                    "audio_stream": audio_status,
                    "inference_executor": {"status": "running", "pending_tasks": pending},
                },
            }
            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


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


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    session_id = await state.manager.connect(websocket)
    if session_id is None:
        await websocket.close(code=1013, reason="Too many clients")
        logger.warning("Client rejected: connection limit reached")
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
                await websocket.send_text('{"type":"pong"}')
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

def submit_translation(audio_np, target_langs, loop):
    """Run translation in ThreadPoolExecutor and push results to translation_queue."""
    try:
        results = translate_audio(
            audio_np, list(target_langs),
            state.processor, state.model, state.device, state.dtype,
            runtime_config,
            state._perf_metrics, state._perf_lock,
            state._translation_history, state._translation_history_lock,
        )
        if results:
            future = asyncio.run_coroutine_threadsafe(state.translation_queue.put(results), loop)
            future.add_done_callback(lambda f: f.exception() and logger.error(f"Failed to queue translation: {f.exception()}"))
    except Exception as e:
        logger.error(f"Translation Error: {e}")
    finally:
        with state.inference_pending_lock:
            state.inference_pending -= 1


INFERENCE_TIMEOUT = 30  # seconds — log critical if inference exceeds this


def processing_loop(loop):
    MAX_BUFFER_SAMPLES = 48000 * 30
    audio_buffer = np.zeros(MAX_BUFFER_SAMPLES, dtype=np.float32)
    buffer_pos = 0
    prev_audio = np.array([], dtype=np.float32)
    silence_start = None
    is_speaking = False
    VAD_MIN_SAMPLES = 512
    last_future = None
    last_future_time = 0

    # Pre-allocated VAD buffer (numpy) with index pointer — no torch.cat per chunk
    VAD_BUFFER_MAX = 16000  # ~1s at 16kHz, far more than needed
    vad_np_buffer = np.zeros(VAD_BUFFER_MAX, dtype=np.float32)
    vad_pos = 0
    vad_buffer_time = time.time()
    # Pre-allocated tensor for VAD inference (reused every call)
    vad_chunk_tensor = torch.zeros(VAD_MIN_SAMPLES, dtype=torch.float32)

    _level_update_interval = 0.1  # update audio level at most 10x/s
    _last_level_update = 0.0
    _pending_level = -60.0
    _pending_peak = -60.0

    while not state.stop_event.is_set():
        try:
            chunk = state.audio_queue.get(timeout=0.1)
            chunk_np = np.concatenate(chunk).flatten()

            # Grab resampler reference under lock (restart_audio_stream writes it)
            with state.audio_stream_lock:
                current_resampler = state.resampler
            chunk_np = resample_audio(chunk_np, current_resampler)

            level = compute_audio_level(chunk_np)
            # Track max since last flush (lock-free local vars)
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

            n = chunk_np.shape[0]
            if buffer_pos + n <= MAX_BUFFER_SAMPLES:
                audio_buffer[buffer_pos:buffer_pos + n] = chunk_np
                buffer_pos += n

            # Append to VAD numpy buffer (no allocation)
            vad_n = min(n, VAD_BUFFER_MAX - vad_pos)
            if vad_n > 0:
                vad_np_buffer[vad_pos:vad_pos + vad_n] = chunk_np[:vad_n]
                vad_pos += vad_n
            vad_buffer_time = time.time()
            speech_detected = is_speaking

            # Process VAD in 512-sample steps using pointer into pre-allocated buffer
            vad_read_pos = 0
            while vad_pos - vad_read_pos >= VAD_MIN_SAMPLES:
                vad_chunk_tensor.copy_(torch.from_numpy(vad_np_buffer[vad_read_pos:vad_read_pos + VAD_MIN_SAMPLES]))
                vad_read_pos += VAD_MIN_SAMPLES
                confidence = state.vad_model(vad_chunk_tensor, runtime_config["sample_rate"]).item()
                speech_detected = confidence > 0.5
            # Shift remaining samples to front (memmove, no allocation)
            remaining = vad_pos - vad_read_pos
            if remaining > 0 and vad_read_pos > 0:
                vad_np_buffer[:remaining] = vad_np_buffer[vad_read_pos:vad_read_pos + remaining]
            vad_pos = remaining

            sr = runtime_config["sample_rate"]
            total_frames = buffer_pos
            total_duration = total_frames / sr
            current_time = time.time()

            if speech_detected:
                is_speaking = True
                silence_start = None
            else:
                if silence_start is None:
                    silence_start = current_time

            silence_dur = runtime_config["silence_duration"]
            max_chunk = runtime_config["max_chunk_duration"]
            min_chunk = runtime_config["min_chunk_duration"]
            ctx_overlap = runtime_config["context_overlap"]

            should_process = False
            if is_speaking and silence_start and (current_time - silence_start > silence_dur):
                should_process = True
            if total_duration > max_chunk:
                should_process = True

            if should_process and total_duration >= min_chunk:
                full_audio = audio_buffer[:buffer_pos].copy().astype(np.float32)

                audio_to_process = full_audio
                if prev_audio.size > 0:
                    audio_to_process = np.concatenate((prev_audio, full_audio))

                target_langs = state.manager.get_unique_languages()
                if target_langs:
                    # Check for stuck inference
                    if last_future is not None and not last_future.done():
                        elapsed = time.time() - last_future_time
                        if elapsed > INFERENCE_TIMEOUT:
                            logger.critical(f"Inference stuck for {elapsed:.0f}s — skipping new submission")
                            last_future.cancel()
                            last_future = None
                            # Skip this chunk
                        else:
                            logger.warning(f"Skipping chunk ({total_duration:.1f}s) — inference busy ({elapsed:.0f}s)")
                    else:
                        with state.inference_pending_lock:
                            if state.inference_pending >= 1:
                                logger.warning(f"Skipping chunk ({total_duration:.1f}s) — inference queue busy")
                            else:
                                state.inference_pending += 1
                                last_future = state.inference_executor.submit(submit_translation, audio_to_process, target_langs, loop)
                                last_future_time = time.time()

                samples_ctx = int(sr * ctx_overlap)
                if full_audio.shape[0] > samples_ctx:
                    prev_audio = full_audio[-samples_ctx:]
                else:
                    prev_audio = full_audio

                buffer_pos = 0
                is_speaking = False
                silence_start = None

            elif not is_speaking and total_duration > min_chunk:
                samples_to_keep = int(sr * 1.0)
                if total_frames > samples_to_keep * 5:
                    buffer_pos = 0

        except queue.Empty:
            # Reset VAD buffer after 5s without new audio data
            if vad_pos > 0 and time.time() - vad_buffer_time > 5.0:
                vad_pos = 0
            continue
        except Exception as e:
            logger.error(f"Loop Error: {e}")

    # Flush remaining buffer on shutdown if enough data
    sr = runtime_config["sample_rate"]
    min_chunk = runtime_config["min_chunk_duration"]
    if buffer_pos > 0 and buffer_pos / sr >= min_chunk:
        full_audio = audio_buffer[:buffer_pos].copy().astype(np.float32)
        if prev_audio.size > 0:
            full_audio = np.concatenate((prev_audio, full_audio))
        target_langs = state.manager.get_unique_languages()
        if target_langs:
            try:
                with state.inference_pending_lock:
                    state.inference_pending += 1
                submit_translation(full_audio, target_langs, loop)
            except Exception as e:
                logger.error(f"Final flush translation error: {e}")


async def broadcaster():
    """Send translated texts to clients per their language."""
    while not state.stop_event.is_set():
        try:
            results = await asyncio.wait_for(state.translation_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            continue
        tasks = []
        for lang, text in results.items():
            logger.info(f"[{lang}] {text}")
            tasks.append(state.manager.send_to_language(lang, text))
        await asyncio.gather(*tasks)


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
                device_idx = runtime_config["audio_device_index"]
                restart_audio_stream(device_idx, runtime_config["audio_channel"], state)
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

    device_idx = runtime_config["audio_device_index"]
    try:
        restart_audio_stream(device_idx, runtime_config["audio_channel"], state)
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
        save_config()

        # 6. Release GPU memory
        if state.device and state.device.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        logger.info("Shutdown complete.")

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8888"))
    logger.info(f"Starting Web Server on {host}:{port}...")
    uvicorn.run(app, host=host, port=port, log_level="info")
