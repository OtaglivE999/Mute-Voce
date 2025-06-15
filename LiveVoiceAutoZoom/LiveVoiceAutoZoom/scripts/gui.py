
import tkinter as tk
from tkinter import messagebox
import numpy as np
import threading
import time
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from recorder import record_audio, save_audio, find_zoom_input
from vad_enhancer import detect_voiced, enhance_audio
from speaker_recognition import extract_embedding, load_known_speakers, recognize_speaker

class VoiceRecorderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Voice Recorder")
        self.root.geometry("800x600")

        self.plot_fig, self.ax = plt.subplots(figsize=(8, 3))
        self.canvas = FigureCanvasTkAgg(self.plot_fig, master=self.root)
        self.canvas.get_tk_widget().pack(pady=20)

        self.speaker_label = tk.Label(root, text="Detected Speaker: ---", font=("Arial", 14))
        self.speaker_label.pack()

        self.log = tk.Text(root, height=10, width=100)
        self.log.pack(pady=10)

        self.recording = False
        self.stream = None
        self.device_index = find_zoom_input()

        self.start_button = tk.Button(root, text="Start Recording", command=self.start_recording)
        self.start_button.pack(side=tk.LEFT, padx=20)

        self.stop_button = tk.Button(root, text="Stop", command=self.stop_recording)
        self.stop_button.pack(side=tk.RIGHT, padx=20)

    def start_recording(self):
        self.recording = True
        threading.Thread(target=self.record_stream).start()

    def stop_recording(self):
        self.recording = False
        self.log.insert(tk.END, "[INFO] Recording stopped.\n")

    def record_stream(self):
        buffer = []
        known_speakers = load_known_speakers()
        self.log.insert(tk.END, "[INFO] Recording started...\n")

        def callback(indata, frames, time_info, status):
            if not self.recording:
                raise sd.CallbackStop()
            mono = indata[:, 0]
            buffer.extend(mono)
            self.ax.clear()
            self.ax.plot(mono)
            self.ax.set_title("Live Waveform")
            self.canvas.draw()

        with sd.InputStream(device=self.device_index, channels=1,
                            samplerate=44100, dtype='int16', callback=callback):
            while self.recording:
                time.sleep(0.1)

        audio_np = np.array(buffer, dtype=np.int16)
        save_audio("scripts/temp_raw.wav", audio_np)

        voiced = detect_voiced(audio_np)
        enhanced = enhance_audio(voiced)
        save_audio("scripts/temp_enhanced.wav", enhanced)

        emb = extract_embedding("scripts/temp_enhanced.wav")
        match = recognize_speaker(emb, known_speakers)
        if match:
            self.speaker_label.config(text=f"Detected Speaker: {match}")
            self.log.insert(tk.END, f"[MATCH] Speaker identified as: {match}\n")
        else:
            self.speaker_label.config(text="Detected Speaker: Unknown")
            self.log.insert(tk.END, "[UNKNOWN] Speaker not recognized\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceRecorderGUI(root)
    root.mainloop()
