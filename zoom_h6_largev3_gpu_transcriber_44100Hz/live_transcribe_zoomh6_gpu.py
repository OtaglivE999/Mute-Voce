import sounddevice as sd
import numpy as np
import queue
import threading
import time
import traceback
from faster_whisper import WhisperModel

# ------------------ CONFIGURATION ------------------
SAMPLERATE = 44100
CHANNELS = 1
CHUNK_DURATION = 5
DEVICE_NAME_FILTER = "zoom"
MODEL_SIZE = "large-v3"
COMPUTE_TYPE = "float16"
DEVICE_TYPE = "cuda"

# ------------------ INITIALIZATION ------------------
audio_queue = queue.Queue()
model = None

def get_zoom_device_index(name_filter):
    try:
        devices = sd.query_devices()
        for i, d in enumerate(devices):
            if name_filter.lower() in d['name'].lower() and d['max_input_channels'] >= CHANNELS:
                print(f"[INFO] Found input device: {d['name']} (index {i})")
                return i
        raise ValueError(f"No input device matching '{name_filter}' found.")
    except Exception as e:
        print(f"[ERROR] Device lookup failed: {e}")
        exit(1)

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"[WARNING] Input stream status: {status}")
    if indata.shape[0] == 0:
        print("[WARNING] No audio data received")
        return
    audio_queue.put(indata.copy())

def transcribe_audio(audio_data):
    try:
        if np.max(np.abs(audio_data)) < 0.01:
            print("[INFO] Skipped silent audio chunk")
            return
        print("[INFO] Transcribing...")
        segments, _ = model.transcribe(
            audio_data,
            language="en",
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False
        )
        for segment in segments:
            print(f"[{segment.start:.2f}s - {segment.end:.2f}s]: {segment.text}")
    except Exception as e:
        print(f"[ERROR] Transcription failed:\n{traceback.format_exc()}")

def capture_audio():
    device_index = get_zoom_device_index(DEVICE_NAME_FILTER)
    try:
        with sd.InputStream(samplerate=SAMPLERATE,
                            channels=CHANNELS,
                            callback=audio_callback,
                            blocksize=int(SAMPLERATE * 0.5),
                            dtype='float32',
                            device=device_index):
            print("[INFO] Audio stream started. Listening...")
            buffer = np.empty((0, CHANNELS), dtype='float32')
            last_time = time.time()
            while True:
                data = audio_queue.get()
                buffer = np.vstack((buffer, data))
                if time.time() - last_time >= CHUNK_DURATION:
                    chunk = buffer[:, 0]
                    threading.Thread(target=transcribe_audio, args=(chunk.copy(),)).start()
                    buffer = np.empty((0, CHANNELS), dtype='float32')
                    last_time = time.time()
    except Exception as e:
        print(f"[FATAL] Failed to open audio stream:\n{traceback.format_exc()}")
        exit(1)

if __name__ == "__main__":
    print("[INFO] Initializing Whisper model...")
    try:
        model = WhisperModel(MODEL_SIZE, device=DEVICE_TYPE, compute_type=COMPUTE_TYPE)
    except Exception as e:
        print(f"[FATAL] Whisper model failed to load:\n{traceback.format_exc()}")
        exit(1)

    try:
        capture_audio()
    except KeyboardInterrupt:
        print("[INFO] Stopped by user.")
    except Exception as e:
        print(f"[FATAL] Unhandled error:\n{traceback.format_exc()}")
