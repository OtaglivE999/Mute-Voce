import sounddevice as sd
import soundfile as sf

print("🎙️ Microphone Test: Recording 5 seconds...")

try:
    fs = 44100
    duration = 5  # seconds
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    sf.write("mic_test_output.wav", audio, fs)
    print("✅ Recording complete. Saved as mic_test_output.wav")
except Exception as e:
    print("❌ Failed to record audio:")
    print(e)
