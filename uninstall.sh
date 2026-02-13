#!/bin/bash

# Ensure script is executable: chmod +x uninstall.sh

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMP_BASE="${TMPDIR:-${TMP:-/tmp}}"
TEMP_BASE="${TEMP_BASE%/}"
VENV_DIR="$TEMP_BASE/venv_preklad_kazani_AI"
HF_CACHE="${HOME}/.cache/huggingface/hub"
TORCH_CACHE="${HOME}/.cache/torch/hub"

echo "=== Live Translation Bridge - Uninstall ==="
echo ""
echo "This will remove:"
echo "  1. Virtual environment ($VENV_DIR)"
echo "  2. Downloaded AI models (~/.cache/huggingface + ~/.cache/torch)"
echo "  3. Generated config file (config.json)"
echo "  4. Python cache (__pycache__)"
echo ""

read -rp "Continue? (y/n): " CONFIRM
if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" ]]; then
    echo "Cancelled."
    exit 0
fi

# 1. Remove virtual environment
if [ -d "$VENV_DIR" ]; then
    echo "Removing virtual environment..."
    rm -rf "$VENV_DIR"
    echo "  Removed: $VENV_DIR"
else
    echo "  Virtual environment not found, skipping."
fi

# 2. Remove HuggingFace model cache
REMOVED_MODELS=0

if [ -d "$HF_CACHE/models--facebook--seamless-m4t-v2-large" ]; then
    echo "Removing SeamlessM4T model cache..."
    rm -rf "$HF_CACHE/models--facebook--seamless-m4t-v2-large"
    echo "  Removed: SeamlessM4T v2 model"
    REMOVED_MODELS=1
fi

# 3. Remove Silero VAD cache (torch hub)
if [ -d "$TORCH_CACHE/snakers4_silero-vad_master" ]; then
    echo "Removing Silero VAD model cache..."
    rm -rf "$TORCH_CACHE/snakers4_silero-vad_master"
    echo "  Removed: Silero VAD model"
    REMOVED_MODELS=1
fi

if [ "$REMOVED_MODELS" -eq 0 ]; then
    echo "  No model caches found, skipping."
fi

# 4. Remove config.json
if [ -f "$BASE_DIR/config.json" ]; then
    echo "Removing config.json..."
    rm -f "$BASE_DIR/config.json"
    echo "  Removed: config.json"
else
    echo "  config.json not found, skipping."
fi

# 5. Remove __pycache__
echo "Removing __pycache__ directories..."
find "$BASE_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
echo "  Done."

echo ""
echo "=== Uninstall complete ==="
echo ""
echo "Note: The application source files were NOT removed."
echo "To fully remove, delete this folder: $BASE_DIR"
