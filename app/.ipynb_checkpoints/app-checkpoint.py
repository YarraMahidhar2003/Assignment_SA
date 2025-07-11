import streamlit as st
import torch
import joblib
import torch.nn.functional as F
from transformers import DistilBertForSequenceClassification

# ================
# 🧠 Load Model + Tokenizer
# ================
@st.cache_resource
def load_model_and_tokenizer():
    # Load tokenizer
    tokenizer = joblib.load("tokenizer.joblib")

    # Load model architecture
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )

    # Load trained weights
    model.load_state_dict(
        torch.load("distilbert_sentiment.pt", map_location=torch.device("cpu"))
    )
    model.eval()
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# ================
# 💬 Streamlit UI
# ================
st.set_page_config(page_title="Sentiment Analysis", layout="centered")
st.title("🧠 Sentiment Analysis on Customer Reviews")
st.write("Paste a review below and see if it's positive or negative!")

user_input = st.text_area("✍️ Enter review text:", height=150)

if st.button("🔍 Analyze Sentiment"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        inputs = tokenizer(
            user_input,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()

        label = "🟢 Positive" if pred == 1 else "🔴 Negative"
        st.markdown(f"### Sentiment: **{label}**")
        st.markdown(f"**Confidence:** `{confidence:.2f}`")