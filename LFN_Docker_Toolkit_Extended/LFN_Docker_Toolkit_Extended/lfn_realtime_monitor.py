import sounddevice as sd
import numpy as np
import queue
import threading
import sqlite3
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from datetime import datetime
import time

SAMPLE_RATE = 44100
DURATION_SEC = 5
LF_RANGE = (20, 100)
HF_RANGE = (20000, 24000)
DB_PATH = "lfn_live_log.db"

monitoring = False
audio_queue = queue.Queue()

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS live_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        lfn_peak REAL,
        lfn_db REAL,
        hf_peak REAL,
        hf_db REAL
    )''')
    conn.commit()
    conn.close()

def analyze_and_plot(audio_data):
    audio_data = audio_data.flatten()
    f, t, Sxx = spectrogram(audio_data, fs=SAMPLE_RATE, nperseg=2048, noverlap=1024)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)

    # LFN
    lfn_mask = (f >= LF_RANGE[0]) & (f <= LF_RANGE[1])
    lfn_freqs = f[lfn_mask]
    lfn_spec = Sxx_db[lfn_mask, :]
    lfn_peak = lfn_freqs[np.unravel_index(np.argmax(lfn_spec), lfn_spec.shape)[0]]
    lfn_db = np.max(lfn_spec)

    # Ultrasonic
    hf_mask = (f >= HF_RANGE[0]) & (f <= HF_RANGE[1])
    hf_freqs = f[hf_mask]
    hf_spec = Sxx_db[hf_mask, :]
    if hf_freqs.size > 0:
        hf_peak = hf_freqs[np.unravel_index(np.argmax(hf_spec), hf_spec.shape)[0]]
        hf_db = np.max(hf_spec)
    else:
        hf_peak, hf_db = 0, -100

    # DB logging
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO live_logs (timestamp, lfn_peak, lfn_db, hf_peak, hf_db) VALUES (?, ?, ?, ?, ?)",
              (datetime.now().isoformat(), lfn_peak, lfn_db, hf_peak, hf_db))
    conn.commit()
    conn.close()

    # Plot
    plt.clf()
    plt.pcolormesh(t, f[f <= 500], Sxx_db[f <= 500, :], shading='gouraud')
    plt.title(f"Live Spectrogram - LFN: {lfn_peak:.1f} Hz @ {lfn_db:.1f} dB | HF: {hf_peak:.1f} Hz @ {hf_db:.1f} dB")
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.pause(0.01)

def audio_callback(indata, frames, time_info, status):
    if monitoring:
        audio_queue.put(indata.copy())

def record_loop(device=None):
    global monitoring
    with sd.InputStream(samplerate=SAMPLE_RATE, device=device, channels=1, callback=audio_callback):
        print("🎙️  Monitoring started (Press ENTER to stop)...")
        plt.ion()
        while monitoring:
            time.sleep(DURATION_SEC)
            buffer = []
            while not audio_queue.empty():
                buffer.append(audio_queue.get())
            if buffer:
                audio_data = np.concatenate(buffer, axis=0)
                analyze_and_plot(audio_data)
        plt.ioff()
        plt.close()

def toggle_monitoring(device=None):
    global monitoring
    if not monitoring:
        monitoring = True
        threading.Thread(target=record_loop, args=(device,), daemon=True).start()
    else:
        monitoring = False
        print("🛑 Monitoring stopped.")

if __name__ == "__main__":
    init_db()
    print("Available audio input devices:")
    print(sd.query_devices())
    selected_device = input("Enter device index or press ENTER for default: ")
    selected_device = int(selected_device) if selected_device.strip().isdigit() else None
    print("Press ENTER to start/stop real-time monitoring. Ctrl+C to exit.")
    try:
        while True:
            input()
            toggle_monitoring(device=selected_device)
    except KeyboardInterrupt:
        monitoring = False
        print("\n[EXIT] Monitoring session ended.")
