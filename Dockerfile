# syntax=docker/dockerfile:1
#
# Single parameterized image for live-translation-bridge.
# The PyTorch flavour is chosen at build time so requirements.txt stays untouched
# and the native (non-Docker) install keeps working exactly as before.
#
#   CPU (default, runs everywhere incl. Win/Mac Docker Desktop):
#       docker build -t live-translation-bridge:cpu .
#
#   GPU (NVIDIA + Linux host, run with --gpus all):
#       docker build --build-arg CUDA=1 -t live-translation-bridge:gpu .
#     optionally pin the exact CUDA wheel index:
#       docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 -t live-translation-bridge:gpu .
#
# No nvidia/cuda base image is needed: the PyTorch CUDA wheels bundle the CUDA
# runtime, so a slim Python base + the NVIDIA Container Toolkit on the host is enough.

FROM python:3.12-slim

# --- Build-time PyTorch selection -------------------------------------------
ARG CUDA=0
ARG TORCH_INDEX_URL=""
# Default CUDA wheel index for the GPU build (verified: torch 2.11.0 ships cu128).
# Override with --build-arg CUDA_INDEX_URL=... for a different CUDA version.
ARG CUDA_INDEX_URL="https://download.pytorch.org/whl/cu128"
ARG TORCH_VERSION=2.11.0
ARG TORCHAUDIO_VERSION=2.11.0
# Set to 1 to bake the SeamlessM4T + Silero VAD models into the image (offline use).
ARG PREFETCH_MODELS=0

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf \
    TORCH_HOME=/models/torch \
    HOST=0.0.0.0 \
    PORT=8888

# --- System runtime dependencies --------------------------------------------
#   libportaudio2 : PortAudio backend used by sounddevice for audio capture
#   libgomp1      : OpenMP runtime required by torch CPU kernels
#   ffmpeg        : torchaudio I/O fallback
#   curl          : used by the container HEALTHCHECK
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libportaudio2 \
        libgomp1 \
        ffmpeg \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- PyTorch (installed first so requirements.txt pins don't override it) ----
RUN set -eux; \
    if [ -n "$TORCH_INDEX_URL" ]; then \
        pip install "torch==${TORCH_VERSION}" "torchaudio==${TORCHAUDIO_VERSION}" --index-url "$TORCH_INDEX_URL"; \
    elif [ "$CUDA" = "1" ]; then \
        pip install "torch==${TORCH_VERSION}" "torchaudio==${TORCHAUDIO_VERSION}" --index-url "$CUDA_INDEX_URL"; \
    else \
        pip install "torch==${TORCH_VERSION}" "torchaudio==${TORCHAUDIO_VERSION}" --index-url https://download.pytorch.org/whl/cpu; \
    fi

# --- Remaining Python dependencies (torch/torchaudio already satisfied) ------
COPY requirements.txt ./
RUN pip install -r requirements.txt

# --- Application code (after deps for better layer caching) ------------------
COPY . .

# --- Optional: pre-download models into the image for air-gapped startup -----
RUN if [ "$PREFETCH_MODELS" = "1" ]; then \
        python -c "from transformers import SeamlessM4Tv2ForSpeechToText, AutoProcessor; AutoProcessor.from_pretrained('facebook/seamless-m4t-v2-large'); SeamlessM4Tv2ForSpeechToText.from_pretrained('facebook/seamless-m4t-v2-large'); import torch; torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True)"; \
    fi

EXPOSE 8888

# Liveness probe: the process and HTTP server are up. /health always returns 200
# while the process lives; use /ready in your orchestrator for full readiness
# (model + audio stream + broadcaster all healthy).
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -fsS http://localhost:8888/health || exit 1

CMD ["python", "app.py"]
