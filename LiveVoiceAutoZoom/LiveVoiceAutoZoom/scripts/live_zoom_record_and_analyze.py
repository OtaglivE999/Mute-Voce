import os
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.io.wavfile import write

def find_zoom_input():
    devices = sd.query_devices()
    for idx, d in enumerate(devices):
        if "H6" in d['name'] or "Zoom" in d['name']:
            if d['max_input_channels'] >= 1:
                return idx
    raise RuntimeError("Zoom H6 input device not found. Ensure it is connected and in Audio Interface mode.")

RECORD_SECONDS = 74 * 60  # 4440 seconds
SAMPLE_RATE = 16000
CHANNELS = 1
SESSION_ID = time.strftime("%Y%m%d_%H%M%S")
RECORD_PATH = f"recordings/session_{SESSION_ID}.wav"
ENHANCE_PATH = f"enhanced/session_{SESSION_ID}_enhanced.wav"
FINGERPRINT_PATH = f"fingerprints/voiceprint_{SESSION_ID}.npy"
LOG_PATH = f"logs/session_{SESSION_ID}.log"

def record_audio(filename, duration, samplerate, channels, device_idx):
    print(f"[+] Recording {duration}s from Zoom H6 (Device {device_idx})...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, device=device_idx)
    sd.wait()
    sf.write(filename, audio, samplerate)
    print(f"[+] Saved to {filename}")
    return filename

def enhance_audio(input_file, output_file):
    data, sr = sf.read(input_file)
    from scipy.signal import butter, lfilter
    def bandpass_filter(data, lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        y = lfilter(b, a, data)
        return y
    filtered_data = bandpass_filter(data, 150, 7000, sr)
    sf.write(output_file, filtered_data, sr)
    print(f"[+] Enhanced audio saved to {output_file}")
    return output_file

def fingerprint_audio(file_path):
    wav = preprocess_wav(file_path)
    encoder = VoiceEncoder()
    embed = encoder.embed_utterance(wav)
    np.save(FINGERPRINT_PATH, embed)
    print(f"[+] Fingerprint saved to {FINGERPRINT_PATH}")
    return embed

def compare_with_existing(embed):
    match_log = []
    for f in os.listdir("fingerprints"):
        if f.endswith(".npy") and SESSION_ID not in f:
            ref = np.load(os.path.join("fingerprints", f))
            similarity = np.inner(embed, ref)
            if similarity > 0.75:
                match_log.append(f"Match with {f}: similarity {similarity:.2f}")
    return match_log

if __name__ == "__main__":
    os.makedirs("recordings", exist_ok=True)
    os.makedirs("enhanced", exist_ok=True)
    os.makedirs("fingerprints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    try:
        device_index = find_zoom_input()
        record_audio(RECORD_PATH, RECORD_SECONDS, SAMPLE_RATE, CHANNELS, device_index)
        enhance_audio(RECORD_PATH, ENHANCE_PATH)
        embedding = fingerprint_audio(ENHANCE_PATH)
        matches = compare_with_existing(embedding)

        with open(LOG_PATH, "w") as f:
            f.write(f"Session: {SESSION_ID}\n")
            f.write("\n".join(matches))
            print(f"[+] Log written to {LOG_PATH}")
    except Exception as e:
        print(f"[ERROR] {e}")
