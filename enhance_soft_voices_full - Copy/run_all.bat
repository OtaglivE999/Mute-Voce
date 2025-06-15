@echo off
cd /d "%~dp0"
echo Installing requirements...
python -m pip install --upgrade pip
python -m pip install librosa soundfile numpy openai-whisper ffmpeg-python --quiet

echo Running script...
python enhance_soft_voices_full.py
pause
