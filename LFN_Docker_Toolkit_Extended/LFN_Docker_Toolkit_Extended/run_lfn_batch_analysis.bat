@echo off
SETLOCAL ENABLEEXTENSIONS

:: Prompt for path to folder containing audio files
set /P "AUDIO_DIR=Enter full path to folder containing audio files (mp3, mp4, wav): "

:: Check if folder exists
IF NOT EXIST "%AUDIO_DIR%" (
    echo ❌ Error: The folder "%AUDIO_DIR%" does not exist.
    pause
    exit /b 1
)

:: Launch the Python script from current directory, passing folder path as argument
echo 🔍 Starting batch analysis on: %AUDIO_DIR%
python lfn_batch_file_analyzer.py "%AUDIO_DIR%"
pause
