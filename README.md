# Live Translation Bridge

Real-time speech translation system that captures audio from a microphone, translates it using Meta's SeamlessM4T v2 model, and displays live subtitles in a web browser. Designed for live events such as sermons, conferences, and multilingual meetings.

## Features

- **Real-time speech-to-text translation** using `facebook/seamless-m4t-v2-large`
- **6 target languages**: Czech, English, Russian, Ukrainian, German, Spanish
- **Per-client language selection** — each viewer picks their own language
- **Silero VAD** for accurate voice activity detection
- **Audio resampling** — works with any device sample rate (auto-resamples to 16kHz)
- **Audio preprocessing** — noise gate, normalization, high-pass filter
- **Admin panel** (`/admin`) — device selection, parameter tuning, VU meter, real-time log with level filtering
- **Persistent configuration** — settings saved to `config.json`, restored on restart with validation
- **Internationalized UI** — viewer in 6 languages, admin panel in Czech/English
- **PWA support** — installable on mobile devices, SW update notifications
- **Dark/Light theme** with auto-detection
- **GPU acceleration** — CUDA, MPS (Apple Silicon), or CPU fallback
- **Optimized inference** — configurable beam search, all-language warmup, minimal audio buffer allocations
- **Dynamic QR code** — auto-generated from server URL for easy mobile access
- **Accessibility** — WCAG AA compliant (focus indicators, contrast ratios, ARIA labels, focus traps)
- **Security** — admin auth, WebSocket auth, rate limiting, CORS, CSP headers, config validation, connection limits
- **Resilience** — graceful shutdown, audio device reconnect, heartbeat ping/pong, reconnect jitter, offline detection
- **Structured logging** — optional JSON log format for production log aggregation (ELK, Datadog)
- **Health check** — `/health` endpoint with 3-tier status (healthy/degraded/unhealthy)

## Requirements

- **Python 3.12+**
- **PortAudio** (for `sounddevice`)
  - Windows: included in `sounddevice` wheels
  - macOS: `brew install portaudio`
  - Linux: `sudo apt-get install libportaudio2`
- **RAM**: 8 GB minimum, 16+ GB recommended
- **GPU** (recommended): NVIDIA with CUDA or Apple Silicon (MPS)

## Installation

```bash
# Create virtual environment
python -m venv venv

# Activate
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Install CUDA PyTorch for GPU acceleration
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Usage

The easiest way to run the application is using the provided launch scripts, which automatically set up a virtual environment and install all dependencies:

**Windows:**
```cmd
run_translation.bat
```

**macOS / Linux:**
```bash
chmod +x run_translation.sh
./run_translation.sh
```

The server starts on `http://0.0.0.0:8888` by default (configurable via `config.env`).

- **Viewer page**: `http://<YOUR_IP>:<PORT>` — select language, view live subtitles
- **Admin panel**: `http://<YOUR_IP>:<PORT>/admin` — configure audio device, tuning parameters, monitor status
- **Health check**: `http://<YOUR_IP>:<PORT>/health` — system health status (no auth required)
- **QR code**: `http://<YOUR_IP>:<PORT>/api/qr.svg` — dynamically generated QR code for the server URL

### Environment Variables

Create a `config.env` file in the project root (you can copy `config.env.example`) or set system environment variables:

```env
ADMIN_USERNAME=admin
ADMIN_PASSWORD=change-me
HOST=0.0.0.0
PORT=8888
```

| Variable | Default | Description |
|----------|---------|-------------|
| `ADMIN_USERNAME` | `admin` | Admin panel login username |
| `ADMIN_PASSWORD` | `admin` | Admin panel login password |
| `CORS_ORIGINS` | *(empty — CORS disabled)* | Comma-separated allowed origins (e.g. `https://example.com`) |
| `MAX_CLIENTS` | `200` | Maximum concurrent WebSocket viewer connections |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8888` | Server port |
| `LOG_FORMAT` | *(empty — human-readable)* | Set to `json` for structured JSON logging |
| `RATE_LIMIT_MAX` | `10` | Max admin API requests per second per IP |
| `ALLOW_INSECURE_ADMIN` | *(empty)* | Set to `1` only for local development if you intentionally want to allow `admin/admin` |
| `ADMIN_WS_TOKEN_TTL` | `3600` | Lifetime in seconds for in-memory admin WebSocket bearer tokens |
| `ADMIN_WS_TOKEN_MAX` | `1000` | Maximum in-memory admin WebSocket bearer tokens |
| `TRANSLATION_QUEUE_MAXSIZE` | `100` | Maximum pending translated result batches before dropping newest overflow |
| `CLIENT_QUEUE_SIZE` | `20` | Maximum pending outbound WebSocket messages per viewer before disconnecting the slow client |
| `CLIENT_SEND_TIMEOUT` | `2.0` | Per-message viewer WebSocket send timeout in seconds |
| `WS_SEND_TIMEOUT` | `2.0` | Admin WebSocket send timeout in seconds |
| `MAX_CLIENTS_PER_IP` | `50` | Maximum concurrent viewer WebSockets from one client IP |
| `MAX_JSON_BODY_BYTES` | `65536` | Maximum JSON body size for admin mutation APIs |
| `WS_MAX_SIZE` | `2048` | Uvicorn maximum WebSocket frame size |
| `LOG_LEVEL` | `INFO` | App logger level |
| `SHUTDOWN_TRANSLATION_FLUSH` | *(empty)* | Set to `1` to translate final buffered audio on shutdown; leave off for bounded production restarts |
| `UVICORN_PROXY_HEADERS` | *(empty)* | Set to `1` only behind a trusted reverse proxy |
| `FORWARDED_ALLOW_IPS` | `127.0.0.1` | Trusted proxy IPs for forwarded headers |

> **Note:** Insecure admin credentials such as `admin/admin` and `admin/change-me` are disabled unless `ALLOW_INSECURE_ADMIN=1` is set. Use a real `ADMIN_PASSWORD` before opening the admin panel.

### Production Rollout

This app is designed as a single-node live audio translator: run one process per audio input and model/GPU instance unless you explicitly split audio capture and inference into separate services.

Before going live:

1. Create `config.env` from `config.env.example`, set a strong `ADMIN_PASSWORD`, and keep `ALLOW_INSECURE_ADMIN=0`.
2. Install dependencies from `requirements.txt` and warm the Hugging Face / Silero caches before the event. Startup should pass without external network access.
3. Run a full startup with the production audio device, GPU and target network: `python app.py`.
4. Confirm `/health` returns component states and `/ready` returns HTTP 200 only after model, broadcaster, processing thread, audio stream and GPU checks are healthy.
5. Run a soak test for at least two hours with expected client count, language mix, reconnects and at least one audio device disconnect/reconnect.

Recommended production defaults are already listed in `config.env.example`. `SHUTDOWN_TRANSLATION_FLUSH=0` is intentional for bounded production restarts; set it to `1` only if preserving the final buffered segment matters more than restart time.

If the app runs behind a trusted reverse proxy, set `UVICORN_PROXY_HEADERS=1` and keep `FORWARDED_ALLOW_IPS` restricted to the proxy IP, for example `127.0.0.1`. The proxy must support WebSocket upgrades for `/ws`, `/api/status/ws`, and `/api/logs`; see `deploy/nginx/live-translation-bridge.conf.example`.

Operational alerts should watch `/ready`, `inference_stuck`, `stuck_inference_count`, queue sizes, GPU memory, and rising drop counters: `audio_queue_drops`, `audio_status_drops`, `audio_buffer_overflows`, `translation_queue_drops`, `ws_messages_dropped`.

### Preview Server

For UI development without ML/audio dependencies:

```bash
python preview_server.py
```

Serves the full frontend with mock data (simulated audio levels, demo subtitles). Uses the same `config.env` for configuration.

### Uninstall

To remove the virtual environment, downloaded AI models, and generated config:

**Windows:**
```cmd
uninstall.bat
```

**macOS / Linux:**
```bash
chmod +x uninstall.sh
./uninstall.sh
```

The scripts will ask for confirmation before deleting anything. Source files are not removed.

## Configuration

All parameters can be adjusted at runtime via the admin panel (`/admin`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| Audio device | System default | Select any input device |
| Audio channel | 0 | Multi-channel device support |
| Silence duration | 0.8s | Pause length to trigger translation |
| Min chunk duration | 1.5s | Minimum audio length for translation |
| Max chunk duration | 12.0s | Force-trigger after this duration |
| Context overlap | 0.5s | Audio overlap between segments |
| Beam Search | 1 | Beam search decoding (1–5). Higher = better quality, slower. See below |
| Noise gate | Off | Silence audio below threshold |
| Noise gate threshold | -40 dB | Threshold level for noise gate |
| Normalization | **On** | RMS-based volume normalization |
| Normalization target | -3 dB | Target RMS level for normalization |
| High-pass filter | **On** | Remove low-frequency rumble (Butterworth 2nd order) |
| High-pass cutoff | 80 Hz | Cutoff frequency for high-pass filter |

### Beam Search

The **Beam Search** parameter controls the decoding strategy of the translation model. It determines the trade-off between translation speed and quality:

| Value | Strategy | Description |
|-------|----------|-------------|
| 1 | Greedy decoding | Picks the most likely token at each step. Fastest, suitable for real-time with low latency |
| 2–3 | Beam search | Explores multiple candidate sequences in parallel. Good balance of quality and speed |
| 4–5 | Wide beam search | Best translation quality, but significantly slower — especially on CPU |

Higher values produce more accurate translations because the model considers more possible sentence continuations before choosing the best one. For real-time use, **1–2** is recommended on GPU, **1** on CPU.

## Supported Languages

| Code | Language |
|------|----------|
| `ces` | Czech (default) |
| `eng` | English |
| `rus` | Russian |
| `ukr` | Ukrainian |
| `deu` | German |
| `spa` | Spanish |

## Architecture

```
Microphone → sounddevice (native SR) → Audio Buffer (native SR)
    → VAD (resampled to 16kHz per chunk) → Segment detection
    → Resample full segment to 16kHz → Preprocessing → SeamlessM4T v2
    → Per-language translation → WebSocket → Browser (live subtitles)
```

- **Backend**: Python, FastAPI, Uvicorn
- **AI Model**: `facebook/seamless-m4t-v2-large` (speech-to-text translation)
- **VAD**: Silero VAD (torch-based)
- **Frontend**: Vanilla JS, CSS3, WebSocket API, PWA
- **Communication**: WebSocket for real-time subtitles, REST API for admin

## Project Structure

```
├── app.py                      # Entry point (initialize + start server)
├── config.env                  # Environment variables (port, credentials, etc.)
├── src/
│   ├── config.py               # Configuration defaults, load/save, validation
│   ├── logging_handler.py      # Log buffer handler, optional JSON formatter
│   ├── state.py                # Global state, locks, initialize()
│   ├── server.py               # FastAPI app, routes, WebSocket, processing loop
│   ├── audio/
│   │   ├── capture.py          # Device detection, VAD, audio stream, resampling
│   │   └── preprocess.py       # High-pass filter, noise gate, normalization
│   └── translation/
│       ├── engine.py           # Model loading, warmup, translate_audio
│       └── session.py          # ClientSession, SessionManager
├── templates/
│   ├── index.html              # Viewer page (subtitles)
│   └── admin.html              # Admin panel
├── static/
│   ├── app.js                  # Viewer frontend (i18n, WebSocket, UI)
│   ├── styles.css              # Viewer styles (WCAG AA)
│   ├── admin.js                # Admin frontend (i18n cs/en)
│   ├── admin.css               # Admin styles
│   ├── sw.js                   # Service Worker (PWA offline cache)
│   ├── manifest.json           # PWA manifest
│   └── assets/                 # Favicon, icons (192/512 PNG)
├── preview_server.py           # Standalone mock server for UI development
├── requirements.txt            # Python dependencies
├── run_translation.bat         # Windows launch script
├── run_translation.sh          # macOS/Linux launch script
├── uninstall.bat               # Windows uninstall script
└── uninstall.sh                # macOS/Linux uninstall script
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is provided as-is for educational and community use.
