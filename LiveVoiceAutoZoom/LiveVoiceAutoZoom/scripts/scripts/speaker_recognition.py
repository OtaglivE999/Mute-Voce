
import os
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine

encoder = VoiceEncoder()
SPEAKER_DB_PATH = "scripts/speakers"  # Folder to store known speaker embeddings
os.makedirs(SPEAKER_DB_PATH, exist_ok=True)

def extract_embedding(wav_path):
    wav = preprocess_wav(wav_path)
    return encoder.embed_utterance(wav)

def save_speaker(name, embedding):
    np.save(os.path.join(SPEAKER_DB_PATH, f"{name}.npy"), embedding)

def load_known_speakers():
    speakers = {}
    for file in os.listdir(SPEAKER_DB_PATH):
        if file.endswith(".npy"):
            name = file.replace(".npy", "")
            embedding = np.load(os.path.join(SPEAKER_DB_PATH, file))
            speakers[name] = embedding
    return speakers

def recognize_speaker(embedding, known_speakers, threshold=0.3):
    for name, ref_embedding in known_speakers.items():
        if cosine(embedding, ref_embedding) < threshold:
            return name
    return None

def cluster_unknown_embeddings(embeddings, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    return labels
