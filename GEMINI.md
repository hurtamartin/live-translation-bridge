# Project Overview

This project is a **real-time speech translation system** designed to capture audio, translate it using a state-of-the-art AI model, and display the translated text on a web interface. It is likely intended for live events or sermons (`preklad_kazani_AI` translates to "sermon translation AI"), providing a subtitle-like experience for viewers.

## Key Technologies

- **Backend:** Python 3.12, FastAPI, Uvicorn.
- **AI/ML:** Hugging Face `transformers` using the `facebook/seamless-m4t-v2-large` model for speech-to-text translation.
- **Audio Processing:** `sounddevice` for audio capture, `numpy` for signal processing (custom VAD - Voice Activity Detection).
- **Communication:** WebSockets for real-time text broadcasting.
- **Frontend:** Jinja2 templates serving a simple, high-contrast HTML/CSS/JS interface.

## Architecture

1.  **Audio Capture:** The system captures audio from a specific input device (default keyword: `<<--Spotify2StudioLive`) or the default system microphone.
2.  **Processing Loop:** A background thread processes audio chunks, detects speech activity (VAD), and triggers the translation model when a sentence is complete or a timeout is reached.
3.  **Translation:** The audio is fed into the `seamless-m4t-v2-large` pipeline, which returns the translated text (default target language: Russian `rus`).
4.  **Broadcasting:** Translated text is sent via an asyncio queue to a broadcaster task, which pushes it to all connected WebSocket clients.
5.  **Display:** The web client (`templates/index.html`) receives the text and displays the last few sentences with a fade-in animation.

# Setup and Usage

## Prerequisites

- **Python 3.12** is required.
- **PortAudio:** Required for `sounddevice`.
  - **Windows:** Usually handled automatically by the `sounddevice` binary wheels.
  - **macOS:** `brew install portaudio` (handled by `run_translation.sh`).
  - **Linux:** `sudo apt-get install libportaudio2` (or similar).

## Installation

1.  Create a virtual environment:
    ```bash
    python -m venv venv
    ```
2.  Activate the environment:
    - **Windows (PowerShell):** `.\venv\Scripts\Activate.ps1`
    - **macOS/Linux:** `source venv/bin/activate`
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

### Windows (PowerShell)
```powershell
# Activate venv
.\venv\Scripts\Activate.ps1

# Run the app
python app.py
```

### macOS / Linux
You can use the provided script:
```bash
./run_translation.sh
```
Or run manually:
```bash
source venv/bin/activate
python app.py
```

The server will start on `http://0.0.0.0:8888`.
Access the interface at `http://localhost:8888` or `http://<YOUR_LOCAL_IP>:8888` from other devices.

# Configuration

Key settings are hardcoded in `app.py` and may need modification:

-   `DEVICE_NAME_KEYWORD`: Substring to match the desired audio input device name.
-   `DEFAULT_TARGET_LANG`: Target language code (default: `"rus"`).
-   `SILENCE_THRESHOLD`, `SILENCE_DURATION`: Parameters for the Voice Activity Detection (VAD).
-   `model="facebook/seamless-m4t-v2-large"`: The Hugging Face model used.

# Development Notes

-   **Concurrency:** The app mixes synchronous code (audio processing, PyTorch inference) with asynchronous code (FastAPI, WebSockets). `asyncio.run_coroutine_threadsafe` is used to bridge the gap.
-   **Performance:** The `seamless-m4t-v2-large` model is computationally intensive. A GPU (CUDA or MPS) is highly recommended for real-time performance.
-   **Frontend:** The frontend is minimal (`index.html`), designed for dark mode/projection. It keeps a history of 5 sentences.
