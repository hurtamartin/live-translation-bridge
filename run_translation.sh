#!/bin/bash

# Ensure script is executable: chmod +x run_translation.sh

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMP_BASE="${TMPDIR:-${TMP:-/tmp}}"
# Remove trailing slash from TEMP_BASE if present to avoid double slashes, though usually harmless
TEMP_BASE="${TEMP_BASE%/}"
VENV_DIR="$TEMP_BASE/venv_preklad_kazani_AI"
PYTHON_CMD="python3.12"

echo "=== Setup Translation Environment ==="

# 1. Check for Python
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo "Error: $PYTHON_CMD could not be found."
    exit 1
fi

# 2. Check/Install PortAudio (Required for sounddevice on Mac)
if ! brew list portaudio &>/dev/null; then
    echo "Installing portaudio via Homebrew (required for audio input)..."
    if command -v brew &> /dev/null; then
        brew install portaudio
    else
        echo "Error: Homebrew not found. Please install 'portaudio' manually."
        exit 1
    fi
else
    echo "PortAudio is installed."
fi

# 3. Virtual Environment Setup
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    $PYTHON_CMD -m venv "$VENV_DIR"
fi

# Activate Venv variables
PYTHON_EXEC="$VENV_DIR/bin/python3"
PIP_EXEC="$VENV_DIR/bin/pip"

# 4. Install Dependencies
echo "Installing dependencies..."
"$PIP_EXEC" install -r "$BASE_DIR/requirements.txt"

# 5. Run Application
echo "Starting Translation Server..."
echo "Open http://<YOUR_IP>:8888 on mobile devices."
cd "$BASE_DIR" || exit
"$PYTHON_EXEC" "app.py"
