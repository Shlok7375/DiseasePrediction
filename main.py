import os
import pickle
import joblib
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import pairwise_distances
from difflib import get_close_matches

# --- PATH SETUP (Git-friendly, relative to script) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH_NB = os.path.join(BASE_DIR, 'trained_model.pkl')
MODEL_PATH_NN = os.path.join(BASE_DIR, 'trained_nn_model.keras')
SYMPTOM_LIST_PATH = os.path.join(BASE_DIR, 'master_symptom_list.pkl')
LABEL_LIST_PATH = os.path.join(BASE_DIR, 'label_list.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'label_encoder.pkl')

# --- LOAD ARTIFACTS ---
with open(SYMPTOM_LIST_PATH, 'rb') as f:
    master_symptoms = pickle.load(f)

with open(LABEL_LIST_PATH, 'rb') as f:
    disease_labels = pickle.load(f)

with open(ENCODER_PATH, 'rb') as f:
    label_encoder = pickle.load(f)

model_nb = joblib.load(MODEL_PATH_NB)
model_nn = tf.keras.models.load_model(MODEL_PATH_NN)

symptom2index = {s: i for i, s in enumerate(master_symptoms)}
num_features = len(master_symptoms)

# --- PARSING + VECTORIZATION ---
def parse_input_sentence(sentence):
    words = sentence.lower().replace(',', ' ').split()
    selected = set()
    for symptom in master_symptoms:
        tokens = symptom.replace('|', ' ').split()
        if any(token in words for token in tokens):
            selected.add(symptom)
        else:
            close = get_close_matches(symptom, words, n=1, cutoff=0.85)
            if close:
                selected.add(symptom)
    return selected

def vectorize_symptoms(symptom_tokens):
    vec = np.zeros(num_features, dtype=np.int8)
    for s in symptom_tokens:
        if s in symptom2index:
            vec[symptom2index[s]] = 1
    return vec.reshape(1, -1)

def show_predictions(vec):
    # Naive Bayes predictions
    nb_probs = model_nb.predict_proba(vec)[0]
    top3_nb = np.argsort(nb_probs)[::-1][:3]
    print("\n🔹 Naive Bayes Predictions:")
    for idx in top3_nb:
        print(f"   {disease_labels[idx]}: {nb_probs[idx]:.4f}")

    # Neural Network predictions
    nn_probs = model_nn.predict(vec, verbose=0)[0]
    top3_nn = np.argsort(nn_probs)[::-1][:3]
    print("\n🔹 Keras NN Predictions:")
    for idx in top3_nn:
        print(f"   {disease_labels[idx]}: {nn_probs[idx]:.4f}")

# --- RUN INTERFACE ---
if __name__ == '__main__':
    print("\n🩺 Enter your symptoms in a single sentence:")
    user_input = input(">> ")

    tokens = parse_input_sentence(user_input)
    if not tokens:
        print("❌ No recognizable symptoms found.")
    else:
        print(f"✅ Matched {len(tokens)} symptoms.")
        vector = vectorize_symptoms(tokens)
        show_predictions(vector)
