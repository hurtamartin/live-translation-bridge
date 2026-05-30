# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Real-time speech translation: captures microphone audio, translates with Meta's SeamlessM4T v2, streams live subtitles to browsers over WebSocket. Each viewer picks its own target language (6 supported: `ces`, `eng`, `rus`, `ukr`, `deu`, `spa`). Single-node design — one process per audio input / GPU instance.

See `README.md` for the full feature list, env var reference, and tuning parameters. This file covers what isn't obvious from a single file.

## Commands

```powershell
# Run the full app (loads model + VAD + warmup, starts server on :8888)
python app.py

# Or use the launch scripts (create venv in %TEMP%, pip install, run)
run_translation.bat          # Windows
./run_translation.sh         # macOS/Linux

# UI development without ML/audio deps — serves frontend with mock data
python preview_server.py

# Tests (unittest, no pytest dependency)
python -m unittest discover -s tests          # all tests
python -m unittest tests.test_session_manager # one module
python -m unittest tests.test_audio_capture.AudioLevelTests.test_empty_audio_is_silence  # one test
```

There is no linter or formatter configured, and no CI. Tests do **not** require the model or audio hardware: `test_audio_capture.py` stubs out `sounddevice`/`torch`/`torchaudio` via `sys.modules` before importing, and `test_session_manager.py` uses a `FakeWebSocket`. When adding tests, keep this stubbing pattern so they run without heavy deps.

## Critical gotchas

- **`load_dotenv("config.env")` must stay first in `app.py`**, before any `src` import. Many modules read `os.environ` at import time (queue sizes, auth creds, timeouts), so env must be loaded first.
- **`initialize()` runs at import in `app.py`** and loads the ~2GB model + VAD + warmup. This is why tests import specific functions rather than whole modules and stub heavy deps.
- **`sample_rate` is fixed at 16000** (`FIXED_CONFIG_VALUES` in `src/config.py`). The VAD and model path are wired around 16kHz. Device native sample rate is separate and handled by resampling — do not try to make model sample rate configurable.
- **Inference is single-worker** (`ThreadPoolExecutor(max_workers=1)`). At most one translation runs at a time and at most one is pending; new audio segments are *skipped* (not queued) while busy, with stuck-inference detection after `INFERENCE_TIMEOUT=30s`.
- **Config is mutated only through `src/config.py` helpers** — `get_config_snapshot()` (returns a copy under `runtime_config_lock`) and `apply_runtime_config_updates()`. Never mutate `runtime_config` directly. Config validation happens both on load (`_validate_loaded_config`, resets out-of-range to defaults) and on every API mutation (`_validate_config_values` in `server.py`).

## Architecture

### Threading + asyncio model (the core)

The data path crosses thread/async boundaries several times. Understanding this is essential before touching the pipeline:

```
sounddevice callback thread        → audio_queue (queue.Queue, drops on full)
processing_loop (own OS thread)    → reads audio_queue, runs VAD, segments speech
  └─ inference_executor (1 worker) → translate_audio(), then
       loop.call_soon_threadsafe → translation_queue (asyncio.Queue)
broadcaster (asyncio task)         → reads translation_queue, send_to_language()
  └─ per-client writer_task        → drains that client's send_queue → WebSocket
```

Supporting tasks/threads, all started in `start_server()`'s `@app.on_event("startup")`:
- `audio_watchdog` (OS thread) — reconnects the audio device with exponential backoff on error.
- `audio_history_ticker` (asyncio task) — samples audio level once/sec for the admin graph.

Key principles:
- **Bounded queues drop rather than block.** Every queue (`audio_queue`, `translation_queue`, per-client `send_queue`) has a maxsize and drops/disconnects on overflow, incrementing a counter in `state._perf_metrics` or `SessionManager._stats`. Audio callbacks and the event loop must never block.
- **Thread→async hand-off is always `loop.call_soon_threadsafe`** (see `submit_translation`). The `loop` reference is captured at startup and passed into `processing_loop`.
- **Shutdown is coordinated via `state.stop_event`** (not daemon threads). `shutdown_event` signals it, cancels async tasks, joins threads with timeouts, then stops audio + releases GPU.

### Sample-rate handling

Audio is buffered at the device's **native** sample rate (no per-chunk resampling of the main buffer). Resampling to 16kHz happens only (1) per-chunk for VAD, and (2) once for the whole segment right before inference. A device switch changes `state.native_sample_rate`, which `processing_loop` detects and resets its buffer. VAD requires exactly 16kHz and is fed in 512-sample steps; VAD RNN state is reset after each segment and every 5 minutes to prevent drift.

### Module layout (`src/`)

- `state.py` — all mutable global state, locks, queues, and the model/VAD singletons. `initialize()` loads them. Other modules `import src.state as state` and read/write through it.
- `config.py` — `DEFAULT_CONFIG`, `CONFIG_RANGES`, `FIXED_CONFIG_VALUES`, `SUPPORTED_LANGUAGES`, load/save (atomic via tmp+replace), snapshot/update helpers.
- `logging_handler.py` — `logger` plus `LogBufferHandler` (500-entry ring buffer + async listeners that feed the `/api/logs` WebSocket). `LOG_FORMAT=json` switches console output to structured JSON.
- `server.py` — FastAPI app, all routes, auth, rate limiting, CSP middleware, the `processing_loop`, `broadcaster`, `audio_watchdog`, and `start_server()`.
- `audio/capture.py` — device detection (`detect_device`: CUDA > MPS > CPU), Silero VAD load, audio level, resampler, and `restart_audio_stream` (builds the sounddevice `InputStream` + callbacks).
- `audio/preprocess.py` — high-pass (cached Butterworth SOS) → noise gate → RMS normalize, applied in `translate_audio`.
- `translation/engine.py` — model load (with `torch.compile` on CUDA), warmup of all 6 languages, and `translate_audio`: **encoder runs once, decoder runs per target language** reusing a shallow `copy()` of the encoder output. Handles GPU OOM by clearing cache and skipping.
- `translation/session.py` — `ClientSession` (one per WebSocket viewer) and `SessionManager` (connect limits per total/IP, language routing, per-client single-writer task, slow-client disconnect on queue overflow).

### Auth

Admin pages/API use HTTP Basic (`ADMIN_USERNAME`/`ADMIN_PASSWORD`). Weak passwords (`admin`, `change-me`, …) are rejected with 503 unless `ALLOW_INSECURE_ADMIN=1`. Browser WebSockets can't send custom headers, so `/admin` issues a short-lived bearer token rendered into the page; admin WS endpoints (`/api/status/ws`, `/api/logs`) accept either the `Authorization` header or `?token=`.

### Frontend (`static/`, `templates/`)

Vanilla JS, no build step. Two independent apps: `app.js` (viewer, `state` object, `TRANSLATIONS` for 6 languages) and `admin.js` (`adminState`, `ADMIN_TRANSLATIONS` for cs/en). PWA via `sw.js` + `manifest.json`. Served by FastAPI from `/static`; HTML rendered through Jinja2 templates.

## Deployment

`config.env` (gitignored) holds runtime config; copy from `config.env.example`. Production notes, `/health` vs `/ready` semantics, reverse-proxy setup, and the operational metrics to alert on are in README's "Production Rollout". Example proxy/service configs live in `deploy/nginx/` and `deploy/systemd/`.

### Docker (optional, parallel to native install)

A single parameterized `Dockerfile` (build arg `CUDA=0|1`) builds CPU or GPU images; `docker-compose.yml` exposes `cpu`/`gpu` profiles, `docker-compose.linux-audio.yml` adds `/dev/snd` passthrough (Linux only). The native `python app.py` workflow is unchanged. Key constraints baked into the design:
- **GPU image needs no `nvidia/cuda` base** — pip CUDA wheels bundle the CUDA runtime; slim base + NVIDIA Container Toolkit on host + `--gpus all` is enough. MPS is unavailable in containers (Mac → CPU only).
- **Microphone passthrough only works on Linux hosts.** On Windows/macOS the Docker Desktop VM cannot reach the host mic, so audio uses the **browser streaming** path: `AUDIO_SOURCE=network` + the `/broadcast` page push 16 kHz PCM into `state.audio_queue` over a WebSocket, bypassing `sounddevice` entirely. `AUDIO_SOURCE` defaults to `device` (current behavior). See README "Docker".
- Models (~2 GB) cache to a named volume at `/models` (`HF_HOME`/`TORCH_HOME`); `PREFETCH_MODELS=1` bakes them in for air-gapped use. `config.env`/`config.json` are never baked in (`.dockerignore`).
