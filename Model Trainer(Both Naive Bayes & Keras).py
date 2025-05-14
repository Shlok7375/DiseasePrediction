#!/usr/bin/env python
"""
TRAINING SCRIPT: Trains both Naive Bayes and Keras NN models on parsed patient data.
Handles large datasets via chunking for Naive Bayes.
"""

# --- IMPORT LIBRARIES ---
import pickle
import numpy as np
import math
from scipy.sparse import csr_matrix, vstack
from sklearn.naive_bayes import BernoulliNB
import joblib
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from sklearn.preprocessing import LabelEncoder

# --- FILE PATHS ---
PARSED_DATA_FILE = 'parsed_data.pkl'  # Adjust file path for local setup or Git
MODEL_OUTPUT_FILE_NB = 'trained_model.pkl'
MODEL_OUTPUT_FILE_NN = 'trained_nn_model.keras'
SYMPTOM_MAP_FILE = 'master_symptom_list.pkl'
LABEL_LIST_FILE = 'label_list.pkl'
LABEL_ENCODER_FILE = 'label_encoder.pkl'

# --- PARAMETERS ---
CHUNK_SIZE = 100_000
TEST_RATIO = 0.02
RANDOM_STATE = 42
MAX_EPOCHS = 50
PATIENCE = 5

# --- SYMPTOM TOKENIZATION ---
def create_composite_token(symptom_record):
    base = symptom_record.get('base_symptom', '').strip().replace('_', ' ').lower()
    if not base:
        return ""
    sub_features = symptom_record.get('sub_features', [])
    cleaned_subs = [feat.strip().replace('_', ' ').lower() for feat in sub_features if feat and not feat.strip().isdigit()]
    cleaned_subs = sorted(cleaned_subs)
    token = base + ("|" + "|".join(cleaned_subs) if cleaned_subs else "")
    return token.replace('||', '|').strip('|')

# --- LOAD DATA ---
print("\n📦 Loading parsed data...")
with open(PARSED_DATA_FILE, 'rb') as f:
    all_patients = pickle.load(f)
if not all_patients:
    raise ValueError("❌ No patient records loaded.")

# --- CREATE FEATURE MAP ---
symptom_set = set()
for p in all_patients:
    for s in p.get('parsed_symptoms', []):
        tok = create_composite_token(s)
        if tok:
            symptom_set.add(tok)

master_symptom_list = sorted(symptom_set)
token2col = {tok: idx for idx, tok in enumerate(master_symptom_list)}
num_features = len(master_symptom_list)
print(f"✅ Feature vector space: {num_features} tokens.")

# Save symptom list immediately
with open(SYMPTOM_MAP_FILE, 'wb') as f:
    pickle.dump(master_symptom_list, f)
print(f"✅ Symptom list saved to {SYMPTOM_MAP_FILE}")

# --- CREATE LABEL ENCODER ---
diseases = sorted(set(p.get('disease', '').strip() for p in all_patients if p.get('disease')))
encoder = LabelEncoder()
encoder.fit(diseases)

# Save label encoder and label list immediately
with open(LABEL_ENCODER_FILE, 'wb') as f:
    pickle.dump(encoder, f)
with open(LABEL_LIST_FILE, 'wb') as f:
    pickle.dump(list(encoder.classes_), f)
print(f"✅ Label encoder and label list saved.")

# --- VECTORIZATION ---
def vectorize(patients):
    X_rows, y_rows = [], []
    for p in patients:
        row = np.zeros(num_features, dtype=np.int8)
        for s in p.get('parsed_symptoms', []):
            tok = create_composite_token(s)
            if tok in token2col:
                row[token2col[tok]] = 1
        label = p.get('disease', '').strip()
        if label:
            X_rows.append(row)
            y_rows.append(label)
    return np.array(X_rows, dtype=np.int8), encoder.transform(y_rows)

# --- SPLIT DATA ---
train_data, test_data = train_test_split(all_patients, test_size=TEST_RATIO, random_state=RANDOM_STATE)
print(f"🧪 Train: {len(train_data)}, Test: {len(test_data)}")

# Vectorize full test set
X_test, y_test = vectorize(test_data)

# --- NAIVE BAYES TRAINING ---
print("\n🧠 Training Naive Bayes in chunks...")
nb_model = BernoulliNB()
is_first = True
for i in range(0, len(train_data), CHUNK_SIZE):
    chunk = train_data[i:i + CHUNK_SIZE]
    X_chunk, y_chunk = vectorize(chunk)
    if is_first:
        nb_model.partial_fit(X_chunk, y_chunk, classes=np.arange(len(encoder.classes_)))
        is_first = False
    else:
        nb_model.partial_fit(X_chunk, y_chunk)

joblib.dump(nb_model, MODEL_OUTPUT_FILE_NB)
print(f"✅ Naive Bayes model saved to {MODEL_OUTPUT_FILE_NB}")

# --- KERAS MODEL ---
def build_model():
    model = keras.Sequential()
    model.add(layers.Input(shape=(num_features,)))  # Input layer with num_features
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(len(encoder.classes_), activation='softmax'))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = build_model()
X_train_full, y_train_full = vectorize(train_data)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=PATIENCE, restore_best_weights=True)
model.fit(X_train_full, y_train_full, validation_split=0.1,
          epochs=MAX_EPOCHS, batch_size=512, callbacks=[early_stop], verbose=2)

model.save(MODEL_OUTPUT_FILE_NN)
print(f"✅ Keras model saved to {MODEL_OUTPUT_FILE_NN}")

# --- EVALUATION ---
nb_preds = nb_model.predict(X_test)
nb_acc = np.mean(nb_preds == y_test)
nn_loss, nn_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n📊 Evaluation Results:")
print(f"   🔹 Naive Bayes Accuracy: {nb_acc:.4f}")
print(f"   🔹 Keras NN Accuracy:    {nn_acc:.4f}")
