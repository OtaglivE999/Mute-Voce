
import numpy as np
import webrtcvad
import collections

SAMPLE_RATE = 44100
FRAME_DURATION = 30  # ms
VAD_MODE = 2  # 0-3: higher = more aggressive

vad = webrtcvad.Vad(VAD_MODE)

def frame_generator(audio, sample_rate, frame_duration_ms):
    frame_len = int(sample_rate * frame_duration_ms / 1000)
    for i in range(0, len(audio) - frame_len + 1, frame_len):
        yield audio[i:i + frame_len]

def detect_voiced(audio, sample_rate=SAMPLE_RATE):
    frames = frame_generator(audio, sample_rate, FRAME_DURATION)
    voiced = []
    for frame in frames:
        if vad.is_speech(frame.tobytes(), sample_rate):
            voiced.extend(frame)
    return np.array(voiced, dtype=np.int16)

def enhance_audio(voiced_audio):
    if len(voiced_audio) == 0:
        return np.zeros(1, dtype=np.int16)
    normalized = voiced_audio / np.max(np.abs(voiced_audio))
    amplified = normalized * 0.9
    return (amplified * 32767).astype(np.int16)
