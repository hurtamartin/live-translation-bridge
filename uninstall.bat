@echo off
setlocal

echo === Live Translation Bridge - Uninstall ===
echo.
echo This will remove:
echo   1. Virtual environment (%TEMP%\venv_preklad_kazani_AI)
echo   2. Downloaded AI models (~\.cache\huggingface + ~\.cache\torch)
echo   3. Generated config file (config.json)
echo   4. Python cache (__pycache__)
echo.

set /p CONFIRM="Continue? (y/n): "
if /i not "%CONFIRM%"=="y" (
    echo Cancelled.
    pause
    exit /b 0
)

set "BASE_DIR=%~dp0"
set "VENV_DIR=%TEMP%\venv_preklad_kazani_AI"
set "HF_CACHE=%USERPROFILE%\.cache\huggingface\hub"
set "TORCH_CACHE=%USERPROFILE%\.cache\torch\hub"

:: 1. Remove virtual environment
if exist "%VENV_DIR%" (
    echo Removing virtual environment...
    rmdir /s /q "%VENV_DIR%"
    echo   Removed: %VENV_DIR%
) else (
    echo   Virtual environment not found, skipping.
)

:: 2. Remove HuggingFace model cache
set "REMOVED_MODELS=0"

if exist "%HF_CACHE%\models--facebook--seamless-m4t-v2-large" (
    echo Removing SeamlessM4T model cache...
    rmdir /s /q "%HF_CACHE%\models--facebook--seamless-m4t-v2-large"
    echo   Removed: SeamlessM4T v2 model
    set "REMOVED_MODELS=1"
)

if exist "%HF_CACHE%\models--facebook--seamless-m4t-v2-large" (
    echo   Warning: Could not fully remove SeamlessM4T cache.
)

:: 3. Remove Silero VAD cache (torch hub)
if exist "%TORCH_CACHE%\snakers4_silero-vad_master" (
    echo Removing Silero VAD model cache...
    rmdir /s /q "%TORCH_CACHE%\snakers4_silero-vad_master"
    echo   Removed: Silero VAD model
    set "REMOVED_MODELS=1"
)

if "%REMOVED_MODELS%"=="0" (
    echo   No model caches found, skipping.
)

:: 4. Remove config.json
if exist "%BASE_DIR%config.json" (
    echo Removing config.json...
    del /f "%BASE_DIR%config.json"
    echo   Removed: config.json
) else (
    echo   config.json not found, skipping.
)

:: 5. Remove __pycache__
echo Removing __pycache__ directories...
for /d /r "%BASE_DIR%" %%d in (__pycache__) do (
    if exist "%%d" (
        rmdir /s /q "%%d"
        echo   Removed: %%d
    )
)

echo.
echo === Uninstall complete ===
echo.
echo Note: The application source files were NOT removed.
echo To fully remove, delete this folder: %BASE_DIR%
echo.
pause
endlocal
