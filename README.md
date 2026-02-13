# 🧠 Emotion Classification using Deep Learning (BiLSTM + TensorFlow)

An end-to-end Natural Language Processing (NLP) project that classifies text into six human emotions using a Bidirectional LSTM model. The project includes data preprocessing, model training, evaluation, and a Streamlit-based web interface for real-time predictions.

---

## 🚀 Features

- Multi-class text classification (6 emotions)
- Deep learning model using Bidirectional LSTM
- Full preprocessing pipeline (cleaning, tokenization, padding)
- Evaluation metrics: Accuracy, F1-score, Confusion Matrix
- Streamlit frontend for real-time predictions
- Modular, production-ready code structure

---

## 🎯 Problem Statement

Understanding human emotions from text is a fundamental challenge in NLP.

Given a short text message, the goal is to classify it into one of the following emotion categories:

| Label | Emotion |
|------|--------|
| 0 | Sadness 😢 |
| 1 | Joy 😊 |
| 2 | Love ❤️ |
| 3 | Anger 😡 |
| 4 | Fear 😨 |
| 5 | Surprise 😲 |

---

## 📊 Dataset

- Source: Twitter-based dataset
- Total Samples: 20,000
- Format: `text;label`

### Example:

i didnt feel humiliated;sadness
this is the best day of my life;joy
i feel uncomfortable;fear


### Split:

- Training set: 16,000 samples
- Validation set: 2,000 samples
- Test set: 2,000 samples

---

## 🧠 Model Architecture

Input Text
↓
Text Cleaning
↓
Tokenization (Keras Tokenizer)
↓
Padding (Fixed Length)
↓
Embedding Layer
↓
Bidirectional LSTM
↓
Dropout (Regularization)
↓
Dense Layer (ReLU)
↓
Softmax Output (6 classes)


---

## ⚙️ Tech Stack

- Python
- TensorFlow / Keras
- NumPy, Pandas
- Scikit-learn
- Matplotlib, Seaborn
- Streamlit

---

## 📁 Create Virtual Environment
python -m venv env
source env/bin/activate   # Linux / Mac
env\Scripts\activate      # Windows

## install dependencies
pip install -r requirements.txt


## install dependencies
python train.py

## 📌 Observations

- Strong performance on **Sadness** and **Joy**
- Moderate confusion between **Love** and **Joy**
- Lower performance on **Surprise** due to class imbalance

---

[Classification Report](metrics_report.png)

## 📈 Evaluation Metrics

The model is evaluated using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **Confusion Matrix**

---

## ⚠️ Challenges

- **Class imbalance** (very few samples for *surprise*)
- **Semantic overlap** between emotions (e.g., *joy vs love*)
- **Short and informal text** (Twitter-based dataset)





