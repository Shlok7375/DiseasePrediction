# DiseasePrediction
A hybrid disease prediction system using Naive Bayes and Keras-based neural network models. Accepts natural language symptom input and returns Top 3 predicted diseases with confidence scores. Achieved 97.17% and 97.51% accuracy, respectively.
# 🔬 Disease Prediction System (Naive Bayes + Keras Neural Network)

This project implements a hybrid disease prediction system that combines a Multinomial Naive Bayes classifier and a Keras-based feedforward neural network. It takes natural language symptom input—including typos and synonyms—and predicts the Top 3 most likely diseases with confidence scores from both models.

## 🧠 Models Used

- **Naive Bayes Classifier**
  - Accuracy: **97.17%**
  - Fast and interpretable

- **Feedforward Neural Network (Keras)**
  - Accuracy: **97.51%**
  - Deep learning-based with dropout and ReLU layers

## 💡 Key Features

- Accepts free-text symptom input
- Handles typographical errors and synonyms
- Predicts Top-3 diseases with confidence
- Evaluates both models side-by-side

## ⚙️ Requirements

```bash
pip install -r requirements.txt
