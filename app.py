import streamlit as st
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

# Load model and tokenizer
model_path = "models/imdb-distilbert"
  # Path to your local model folder
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.eval()

# App title
st.title("🎬 IMDb Movie Review Sentiment Classifier")
st.markdown("Paste a review below and click **Predict** to see if it's positive or negative.")

# User input
review = st.text_area("Enter movie review here:", height=150)

# Predict function
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probs).item()
        label = "Positive 😊" if predicted_class == 1 else "Negative 😞"
        return label, probs.squeeze().tolist()

# Button to predict
if st.button("Predict"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        sentiment, scores = predict_sentiment(review)
        st.success(f"**Predicted Sentiment:** {sentiment}")
        st.markdown(f"**Confidence Scores:** Negative: `{scores[0]:.4f}`, Positive: `{scores[1]:.4f}`")
