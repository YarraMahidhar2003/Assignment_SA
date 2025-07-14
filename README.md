# 🧠 Sentiment Analysis Web App using DistilBERT

A modern, real-time **Sentiment Analysis** application that uses **DistilBERT**, a lighter version of BERT, to classify user input text into **Positive**, **Negative**, or **Neutral** sentiment. The web interface is built using **Streamlit** for quick and interactive usage.

---

## 🚀 Project Overview

This project leverages a pre-trained transformer model (**DistilBERT**) fine-tuned on a sentiment-labeled dataset. It allows users to type in text and instantly receive a prediction of the sentiment category, all via an intuitive web UI.

---

## 🔧 Tech Stack

- **Frontend**: Streamlit (Python-based web UI)
- **Backend**: HuggingFace Transformers (DistilBERT), PyTorch
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Model Saving**: `joblib`, `pickle`
- **Visualization**: SHAP (for explainability)

---

## 🧰 Features

- ✔️ Real-time text sentiment prediction
- ✔️ Transformer-based NLP with DistilBERT
- ✔️ Lightweight and fast performance
- ✔️ Streamlit-powered user interface
- ✔️ SHAP interpretability support (Optional)
- ✔️ Docker-ready (if applicable)

---

## 🧪 Model Details

- **Model**: `distilbert-base-uncased`
- **Training**: Fine-tuned on labeled text sentiment data (e.g., IMDb, Twitter, or custom)
- **Classes**: Positive, Negative, Neutral
- **Evaluation Metrics**: Accuracy, F1 Score

---

## 📦 Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/sentiment-analysis-app.git
   cd sentiment-analysis-app
