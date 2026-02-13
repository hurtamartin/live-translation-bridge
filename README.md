# Live Translation Bridge

Real-time speech translation system that captures audio from a microphone, translates it using Meta's SeamlessM4T v2 model, and displays live subtitles in a web browser. Designed for live events such as sermons, conferences, and multilingual meetings.

## Features

- **Real-time speech-to-text translation** using `facebook/seamless-m4t-v2-large`
- **6 target languages**: Czech, English, Spanish, Ukrainian, German, Polish
- **Per-client language selection** — each viewer picks their own language
- **Silero VAD** for accurate voice activity detection
- **Audio resampling** — works with any device sample rate (auto-resamples to 16kHz)
- **Audio preprocessing** — noise gate, normalization, high-pass filter
- **Admin panel** (`/admin`) — device selection, parameter tuning, VU meter, real-time log
- **Persistent configuration** — settings saved to `config.json`, restored on restart
- **Automatic source language detection** — optional model-based detection
- **Internationalized UI** — frontend adapts to the selected language
- **PWA support** — installable on mobile devices
- **Dark/Light theme** with auto-detection
- **GPU acceleration** — CUDA, MPS (Apple Silicon), or CPU fallback

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

```bash
python app.py
```

The server starts on `http://0.0.0.0:8888`.

- **Viewer page**: `http://localhost:8888` — select language, view live subtitles
- **Admin panel**: `http://localhost:8888/admin` — configure audio device, tuning parameters, monitor status

On Windows, you can also use the provided batch script:

```cmd
run_translation.bat
```

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
| Noise gate | Off | Silence audio below threshold |
| Noise gate threshold | -40 dB | Threshold level for noise gate |
| Normalization | Off | Equalize volume levels |
| Normalization target | -3 dB | Target level for normalization |
| High-pass filter | Off | Remove low-frequency rumble |
| High-pass cutoff | 80 Hz | Cutoff frequency for high-pass filter |
| Auto language detection | Off | Detect source language automatically |

## Supported Languages

| Code | Language |
|------|----------|
| `ces` | Czech (default) |
| `eng` | English |
| `spa` | Spanish |
| `ukr` | Ukrainian |
| `deu` | German |
| `pol` | Polish |

## Architecture

```
Microphone → sounddevice (native SR) → Resample to 16kHz → Silero VAD
    → Audio Buffer → SeamlessM4T v2 → Per-language translation
    → WebSocket → Browser (live subtitles)
```

- **Backend**: Python, FastAPI, Uvicorn
- **AI Model**: `facebook/seamless-m4t-v2-large` (speech-to-text translation)
- **VAD**: Silero VAD (torch-based)
- **Frontend**: Vanilla JS, CSS3, WebSocket API, PWA
- **Communication**: WebSocket for real-time subtitles, REST API for admin

## Project Structure

```
├── app.py                      # Entry point (initialize + start server)
├── src/
│   ├── config.py               # Configuration defaults, load/save, runtime_config
│   ├── logging_handler.py      # Log buffer handler for admin panel
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
│   ├── styles.css              # Viewer styles
│   ├── admin.js                # Admin frontend (i18n cs/en)
│   ├── admin.css               # Admin styles
│   ├── sw.js                   # Service Worker (PWA)
│   ├── manifest.json           # PWA manifest
│   └── assets/                 # Favicon, QR code
├── preview_server.py           # Standalone mock server for UI development
├── requirements.txt            # Python dependencies
├── run_translation.bat         # Windows launch script
└── run_translation.sh          # macOS/Linux launch script
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is provided as-is for educational and community use.
