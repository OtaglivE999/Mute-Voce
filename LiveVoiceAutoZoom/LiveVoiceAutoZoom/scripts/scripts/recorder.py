
import sounddevice as sd
import soundfile as sf
import numpy as np

SAMPLE_RATE = 44100
CHANNELS = 1
BIT_DEPTH = 'PCM_16'

def find_zoom_input():
    for idx, d in enumerate(sd.query_devices()):
        if "Zoom" in d['name'] and d['max_input_channels'] >= 1:
            return idx
    raise RuntimeError("Zoom H6 not found. Connect it and enable Audio Interface mode.")

def record_audio(duration_sec=10, device_index=None):
    print(f"[Recording] {duration_sec}s from Zoom H6...")
    if device_index is None:
        device_index = find_zoom_input()

    audio = sd.rec(int(duration_sec * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=CHANNELS, dtype='int16', device=device_index)
    sd.wait()
    return audio

def save_audio(filename, audio_data):
    sf.write(filename, audio_data, SAMPLE_RATE, subtype=BIT_DEPTH)
    print(f"[Saved] Audio to {filename}")
