# 🧠 Sentiment Analysis App

This is a full-stack sentiment analysis system built with **Streamlit**, using two models:
- **Logistic Regression** with TF-IDF
- **DistilBERT (fine-tuned)** using HuggingFace Transformers

It allows users to paste customer reviews and receive **real-time sentiment predictions** along with **confidence scores**.

---

## 🚀 Features

### ✅ End-User Interface
- Paste or type review text
- Get sentiment (Positive/Negative)
- See prediction confidence
- Model runs in real-time using Streamlit

### 🧪 Model Comparison
- Logistic Regression (TF-IDF)
- DistilBERT (Transformer Fine-tuned)
- Accuracy: ~80% vs ~92%
- Visual performance comparison (confusion matrix, classification report)

### 💾 Saved Assets
- `logreg_model.pkl` – Trained Logistic Regression model
- `tfidf_vectorizer.pkl` – TF-IDF vectorizer
- `distilbert_sentiment.pt` – Fine-tuned DistilBERT model weights
- `tokenizer.joblib` – Tokenizer for DistilBERT

---

## 📂 Project Structure

.
├── app/
│ ├── app.py # Streamlit frontend
│ ├── distilbert_sentiment.pt # Fine-tuned DistilBERT weights
│ ├── tokenizer.joblib # Tokenizer
├── data/
│ └── amazon_reviews.csv # Raw dataset used
├── models/
│ ├── logreg_model.pkl # Logistic Regression model
│ ├── tfidf_vectorizer.pkl # TF-IDF vectorizer
├── notebook/
│ └── model_training.ipynb # Training code & evaluation
├── README.md # Project documentation


---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/your-username/sentiment-analysis-app.git
cd sentiment-analysis-app

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
