import pickle
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    top_k_accuracy_score,
    accuracy_score,
)
from sklearn.preprocessing import LabelEncoder

# === Load Artifacts ===
print("🔍 Loading models and metadata...")
nb_model = joblib.load("trained_model.pkl")
nn_model = tf.keras.models.load_model("trained_nn_model.keras")

with open("label_encoder.pkl", "rb") as f:
    encoder: LabelEncoder = pickle.load(f)

with open("label_list.pkl", "rb") as f:
    label_list = pickle.load(f)

with open("master_symptom_list.pkl", "rb") as f:
    symptom_list = pickle.load(f)

# === Load Test Data ===
print("📦 Loading test data...")
with open("parsed_data.pkl", "rb") as f:
    all_patients = pickle.load(f)

from sklearn.model_selection import train_test_split

token2col = {tok: idx for idx, tok in enumerate(symptom_list)}
num_features = len(symptom_list)

def vectorize(patients):
    X, y = [], []
    for p in patients:
        row = np.zeros(num_features, dtype=np.int8)
        for s in p.get("parsed_symptoms", []):
            base = s.get("base_symptom", "").strip().replace("_", " ").lower()
            sub = s.get("sub_features", [])
            token = base
            if sub:
                sub = sorted([
                    i.strip().replace("_", " ").lower()
                    for i in sub if i and not i.strip().isdigit()
                ])
                token += "|" + "|".join(sub)
            token = token.replace("||", "|").strip("|")
            if token in token2col:
                row[token2col[token]] = 1
        label = p.get("disease", "").strip()
        if label:
            X.append(row)
            y.append(label)
    return np.array(X), encoder.transform(y)

train_data, test_data = train_test_split(all_patients, test_size=0.02, random_state=42)
X_test, y_test = vectorize(test_data)
# === Naive Bayes Evaluation ===
print("\n📊 Naive Bayes Evaluation:")
y_pred_nb = nb_model.predict(X_test)
y_prob_nb = nb_model.predict_proba(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb, target_names=encoder.classes_))

print("\nTop-K Accuracy:")
for k in [1, 3, 5, 10]:
    score = top_k_accuracy_score(y_test, y_prob_nb, k=k, labels=np.arange(len(label_list)))
    print(f"Top-{k} Accuracy: {score:.4f}")

# === Confusion Matrix (Naive Bayes) ===
cm_nb = confusion_matrix(y_test, y_pred_nb)
plt.figure(figsize=(14, 12))
sns.heatmap(cm_nb, xticklabels=encoder.classes_, yticklabels=encoder.classes_, annot=False, cmap="Blues")
plt.title("Confusion Matrix (Naive Bayes)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()
# === Predict Keras===
print("🔮 Running model predictions...")
y_prob = model.predict(X_test, batch_size=512, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

# === Metrics ===
print("\n📊 Evaluation Metrics:")
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("✅ Top-K Accuracy:")
for k in [1, 3, 5, 10]:
    acc = top_k_accuracy_score(y_test, y_prob, k=k, labels=np.arange(len(label_list)))
    print(f"   - Top-{k} Accuracy: {acc:.4f}")

# === Classification Report ===
print("\n📄 Classification Report:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, xticklabels=encoder.classes_, yticklabels=encoder.classes_, cmap="Greens", annot=False)
plt.title("Confusion Matrix - Keras Model")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# === Optional: Plot Training History ===
try:
    with open("training_history.pkl", "rb") as f:
        history = pickle.load(f)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["accuracy"], label="Train")
    plt.plot(history["val_accuracy"], label="Val")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["loss"], label="Train")
    plt.plot(history["val_loss"], label="Val")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"⚠️ Training history unavailable or not saved. Skipping plot.")
