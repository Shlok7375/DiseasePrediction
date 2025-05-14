# 🩺 Disease Prediction System (Keras Neural Network)

This project implements a disease prediction system using a Keras-based neural network trained on a structured binary symptom dataset. The model accepts natural language symptom input, including typos and synonyms, and returns the top 3 most probable diseases with confidence scores. The system includes preprocessing, evaluation, and visualization tools(confusion matrices) to support deployment and research use.

---

## 🔍 Key Features

- Natural language symptom input
- Fuzzy matching and typo tolerance (via sentence-transformers + fallback)
- Binary vectorization (645 symptoms)
- Deep learning model using Keras
- Top-3 ranked disease predictions with softmax probabilities
- Full evaluation pipeline with metrics and confusion matrices

---

## 🧠 Model Overview

### ✅ Feedforward Neural Network
- Input size: 645 features (binary symptom indicators)
- Hidden layers: [512 units + dropout] → [256 units + dropout]
- Output: 100 softmax disease classes
- Trained with categorical cross-entropy, Adam optimizer

---

## 🗃️ Repository Structure & Key Files

| File                                            | Purpose                                                                |
|-------------------------------------------------|------------------------------------------------------------------------|
| `main.py`                                       | CLI script to enter symptoms and get predictions from the trained model|
| `Model Trainer (Both naive bayes and keras).py` |Training script for Naive Bayes + Keras                                 |
| `trained_nn_model.keras`                        | Saved Keras neural network model                                       |
| `evaluate_nn_model.py`                          | Evaluation script with metrics, confusion matrix, and plots            |
| `labelencoder.pkl`                              | scikit-learn LabelEncoder mapping disease names to class indices       |
| `label_list.pkl`                                | Ordered list of class label names                                      |
| `master_symptom_list.pkl`                       | Master list of 645 symptom tokens used in training                     |
| `requirements.txt`                              | Required Python libraries                                              |
| `confusion_matrix.png`                          | Visual output showing per-class performance of the neural network      |
| `training_history.pkl`                          | Serialized accuracy/loss per epoch (for plotting)                      |
 
> 📌 **Note**: The dataset (`parsed_data.pkl`) is not included in this repo for size/privacy reasons.

---

## ⚙️ Installation

Ensure Python 3.10+ is installed. Then run:

```bash
pip install -r requirements.txt
