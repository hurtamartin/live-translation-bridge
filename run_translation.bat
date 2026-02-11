@echo off
setlocal

set "BASE_DIR=%~dp0"
set "VENV_DIR=%TEMP%\venv_preklad_kazani_AI"
set "PYTHON_CMD=python"

echo === Setup Translation Environment ===

:: 1. Check for Python
%PYTHON_CMD% --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python could not be found. Please install Python 3.12+ and add it to PATH.
    pause
    exit /b 1
)

:: 2. PortAudio note (not needed on Windows for sounddevice)
echo PortAudio: On Windows, sounddevice includes PortAudio automatically.

:: 3. Virtual Environment Setup
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Creating virtual environment in %VENV_DIR%...
    %PYTHON_CMD% -m venv "%VENV_DIR%"
)

set "PYTHON_EXEC=%VENV_DIR%\Scripts\python.exe"
set "PIP_EXEC=%VENV_DIR%\Scripts\pip.exe"

:: 4. Install Dependencies
echo Installing dependencies...
"%PIP_EXEC%" install -r "%BASE_DIR%requirements.txt"

:: 5. Run Application
echo Starting Translation Server...
echo Open http://YOUR_IP:8888 on mobile devices.
cd /d "%BASE_DIR%"
"%PYTHON_EXEC%" "app.py"

pause
endlocal
