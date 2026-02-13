#!/bin/bash

# Ensure script is executable: chmod +x run_translation.sh

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMP_BASE="${TMPDIR:-${TMP:-/tmp}}"
TEMP_BASE="${TEMP_BASE%/}"
VENV_DIR="$TEMP_BASE/venv_preklad_kazani_AI"

echo "=== Setup Translation Environment ==="

# 1. Find Python 3.12+
PYTHON_CMD=""
for cmd in python3.13 python3.12 python3; do
    if command -v "$cmd" &> /dev/null; then
        version=$("$cmd" -c "import sys; print(sys.version_info[:2])" 2>/dev/null)
        major=$("$cmd" -c "import sys; print(sys.version_info[0])" 2>/dev/null)
        minor=$("$cmd" -c "import sys; print(sys.version_info[1])" 2>/dev/null)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 12 ] 2>/dev/null; then
            PYTHON_CMD="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "Error: Python 3.12+ could not be found. Please install Python 3.12 or newer."
    exit 1
fi
echo "Using Python: $PYTHON_CMD ($($PYTHON_CMD --version))"

# 2. Check/Install PortAudio (required for sounddevice)
if [[ "$(uname)" == "Darwin" ]]; then
    # macOS
    if ! brew list portaudio &>/dev/null 2>&1; then
        echo "Installing portaudio via Homebrew (required for audio input)..."
        if command -v brew &> /dev/null; then
            brew install portaudio
        else
            echo "Error: Homebrew not found. Install portaudio manually: https://brew.sh"
            exit 1
        fi
    else
        echo "PortAudio is installed."
    fi
else
    # Linux
    if ! ldconfig -p 2>/dev/null | grep -q libportaudio; then
        echo "Installing portaudio (required for audio input)..."
        if command -v apt-get &> /dev/null; then
            sudo apt-get update && sudo apt-get install -y libportaudio2 portaudio19-dev
        elif command -v dnf &> /dev/null; then
            sudo dnf install -y portaudio portaudio-devel
        elif command -v pacman &> /dev/null; then
            sudo pacman -S --noconfirm portaudio
        else
            echo "Error: Could not detect package manager. Install portaudio manually."
            exit 1
        fi
    else
        echo "PortAudio is installed."
    fi
fi

# 3. Virtual Environment Setup
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    $PYTHON_CMD -m venv "$VENV_DIR"
fi

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
