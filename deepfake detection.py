import tkinter as tk
from tkinter import filedialog, messagebox
import os
import librosa
import numpy as np
import joblib
from pydub import AudioSegment
import threading

# Load the trained model
MODEL_FILENAME = "best_model_RandomForest.pkl"
model = joblib.load(MODEL_FILENAME)

# Convert to WAV if needed
def convert_to_wav(file_path):
    if file_path.endswith(".wav"):
        return file_path
    wav_path = file_path.rsplit(".", 1)[0] + "_converted.wav"
    audio = AudioSegment.from_file(file_path)
    audio.export(wav_path, format="wav")
    return wav_path

# Feature extraction (analyzes first 5 sec only)
def extract_features(file_path, n_mfcc=13, max_duration=5):
    audio, sr = librosa.load(file_path, sr=None, duration=max_duration)
    if len(audio) < 2048:
        raise ValueError("Audio too short.")
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0).reshape(1, -1)

# Threaded prediction logic
def predict_audio(file_path):
    def run():
        try:
            update_status("🔄 Analyzing...", "orange")
            disable_buttons()
            wav_path = convert_to_wav(file_path)
            features = extract_features(wav_path)
            prediction = model.predict(features)[0]

            label = "🟢 REAL" if prediction == 0 else "🔴 FAKE"
            update_status(f"Prediction: {label}", "#00ffcc" if prediction == 0 else "#ff4444")

        except Exception as e:
            update_status("⚠️ Error", "red")
            messagebox.showerror("Prediction Error", str(e))
        finally:
            enable_buttons()

    threading.Thread(target=run).start()

# Select and analyze file
def select_audio():
    file_path = filedialog.askopenfilename(title="Choose an audio file")
    if file_path:
        predict_audio(file_path)

# Status + button helpers
def update_status(msg, color="white"):
    result_label.config(text=msg, fg=color)

def disable_buttons():
    select_btn.config(state="disabled")

def enable_buttons():
    select_btn.config(state="normal")

# GUI setup
app = tk.Tk()
app.title("Deepfake Audio Detector")
app.geometry("450x300")
app.configure(bg="#1e1e1e")

tk.Label(app, text="🎧 Deepfake Audio Detector", font=("Helvetica", 16, "bold"), fg="#00ffcc", bg="#1e1e1e").pack(pady=30)

select_btn = tk.Button(app, text="📂 Select Audio File", command=select_audio,
                       font=("Helvetica", 12), bg="#00ffcc", fg="black", padx=10, pady=5)
select_btn.pack(pady=20)

result_label = tk.Label(app, text="", font=("Helvetica", 14, "bold"), bg="#1e1e1e", fg="white")
result_label.pack(pady=30)

app.mainloop()
