import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import soundfile as sf
import sys

LF_RANGE = (20, 100)
HF_RANGE = (20000, 24000)
OUTPUT_CSV = "lfn_analysis_results.csv"
SPECTROGRAM_FOLDER = "spectrograms"

os.makedirs(SPECTROGRAM_FOLDER, exist_ok=True)
results = []

def convert_to_wav(input_path, output_path):
    command = ["ffmpeg", "-y", "-i", input_path, "-ar", "44100", "-ac", "1", output_path]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def analyze_audio(filepath, label):
    data, sr = sf.read(filepath)
    if data.ndim > 1:
        data = data.mean(axis=1)

    f, t, Sxx = spectrogram(data, fs=sr, nperseg=4096, noverlap=2048)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)

    lfn_mask = (f >= LF_RANGE[0]) & (f <= LF_RANGE[1])
    lfn_freqs = f[lfn_mask]
    lfn_spec = Sxx_db[lfn_mask, :]
    lfn_peak = lfn_freqs[np.unravel_index(np.argmax(lfn_spec), lfn_spec.shape)[0]]
    lfn_db = np.max(lfn_spec)

    hf_mask = (f >= HF_RANGE[0]) & (f <= HF_RANGE[1])
    hf_freqs = f[hf_mask]
    hf_spec = Sxx_db[hf_mask, :]
    if hf_freqs.size > 0:
        hf_peak = hf_freqs[np.unravel_index(np.argmax(hf_spec), hf_spec.shape)[0]]
        hf_db = np.max(hf_spec)
    else:
        hf_peak, hf_db = 0, -100

    plt.figure(figsize=(12, 6))
    plt.pcolormesh(t, f[f <= 500], Sxx_db[f <= 500, :], shading='gouraud')
    plt.title(f"Spectrogram: {label}")
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.colorbar(label="Intensity (dB)")
    out_img = os.path.join(SPECTROGRAM_FOLDER, f"{os.path.splitext(label)[0]}.png")
    plt.tight_layout()
    plt.savefig(out_img)
    plt.close()

    return {
        "Filename": label,
        "LFN Peak (Hz)": round(float(lfn_peak), 2),
        "LFN dB": round(float(lfn_db), 2),
        "Ultrasonic Peak (Hz)": round(float(hf_peak), 2),
        "Ultrasonic dB": round(float(hf_db), 2),
        "Spectrogram": out_img
    }

def main():
    if len(sys.argv) < 2:
        print("❌ Please provide the path to the audio directory.")
        return
    input_dir = sys.argv[1]
    for file in os.listdir(input_dir):
        ext = file.lower().split('.')[-1]
        if ext in ["wav", "mp3", "mp4"]:
            full_path = os.path.join(input_dir, file)
            label = os.path.splitext(file)[0]
            wav_path = full_path
            if ext != "wav":
                wav_path = os.path.join(input_dir, f"{label}_converted.wav")
                convert_to_wav(full_path, wav_path)
            print(f"Analyzing {file}...")
            try:
                result = analyze_audio(wav_path, file)
                results.append(result)
            except Exception as e:
                print(f"Error analyzing {file}: {e}")

    df = pd.DataFrame(results)
    out_csv = os.path.join(input_dir, OUTPUT_CSV)
    df.to_csv(out_csv, index=False)
    print(f"\n✅ Analysis complete. Results saved to {out_csv} and {SPECTROGRAM_FOLDER}/")

if __name__ == "__main__":
    main()
