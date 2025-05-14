# 🩺 Disease Prediction System (Naive Bayes + Keras Neural Network)

This project implements a hybrid disease prediction system that combines a Multinomial Naive Bayes classifier and a Keras-based feedforward neural network to identify the top 3 most probable diseases based on user-supplied symptoms. The system handles free-text input with potential typographical errors and synonyms, and returns ranked predictions from both models with confidence scores.

---

## 🔍 Key Features

- Accepts natural language symptom input
- Fuzzy matching and typo handling using sentence embeddings + fallback logic
- Binary symptom vectorization (645 features)
- Dual-model prediction (Naive Bayes + Neural Network)
- Top-3 ranked predictions with class probabilities
- Robust preprocessing and vector generation
- Clean CLI-based interaction via `main.py`

---

## 🧠 Models Used

### 1. **Multinomial Naive Bayes**
- File: `trained_model.pkl`
- Format: scikit-learn model
- Description: Probabilistic classifier trained on binary symptom vectors; interpretable and fast.

### 2. **Feedforward Neural Network (Keras)**
- File: `trained_nn_model.keras`
- Format: Keras `.keras` saved model
- Description: Deep neural network with two hidden layers trained using softmax and categorical crossentropy loss.

---

## 🗃️ Supporting Files and Their Roles

| File                                           | Purpose                                                                 |
|------------------------------------------------|-------------------------------------------------------------------------|
| `main.py`                                      | The main runtime script. Accepts input and outputs predictions.         |
| `trained_model.pkl`                            | Trained Naive Bayes model.                                              |
| `trained_nn_model.keras`                       | Trained Keras neural network model.                                     |
| `master_symptom_list.pkl`                      | List of 645 unique, tokenized symptoms. Used for vector creation.       |
| `labelencoder.pkl`                             | Encodes string class labels to integers during training/inference.      |
| `label_list.pkl`                               | List of original disease names (labels), used to decode predictions.    |
| `Model Trainer(Both Naive Bayes & Keras)`|     | Trains both models                                                      |
| `requirements.txt`                             | Python dependencies required to run the project.                        |

---

## 📁 Recommended Directory Structure


