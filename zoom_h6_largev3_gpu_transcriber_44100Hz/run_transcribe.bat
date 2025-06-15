@echo off
SETLOCAL

:: Auto-install required Python packages
echo Installing required Python packages...
python -m pip install --quiet --upgrade pip
python -m pip install --quiet faster-whisper sounddevice numpy

:: Change to script directory
CD /D "%~dp0"

:: Run the transcription script
echo Running Zoom H6 Live Transcription on GPU with large-v3...
python live_transcribe_zoomh6_gpu.py

pause
ENDLOCAL
