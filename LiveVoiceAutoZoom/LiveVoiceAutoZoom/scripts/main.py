
import sys

def run_gui():
    from gui import VoiceRecorderGUI
    import tkinter as tk
    root = tk.Tk()
    app = VoiceRecorderGUI(root)
    root.mainloop()

def run_cli():
    from recorder import record_audio, save_audio
    from vad_enhancer import detect_voiced, enhance_audio
    from speaker_recognition import extract_embedding, load_known_speakers, recognize_speaker

    print("[CLI MODE] Recording 10 seconds...")
    audio = record_audio(10)
    save_audio("scripts/temp_raw.wav", audio)

    voiced = detect_voiced(audio)
    enhanced = enhance_audio(voiced)
    save_audio("scripts/temp_enhanced.wav", enhanced)

    emb = extract_embedding("scripts/temp_enhanced.wav")
    speakers = load_known_speakers()
    match = recognize_speaker(emb, speakers)

    if match:
        print(f"[MATCH] Speaker identified as: {match}")
    else:
        print("[UNKNOWN] Speaker not recognized.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        run_cli()
    else:
        run_gui()
