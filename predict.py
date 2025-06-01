from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import os

# Set the model path (adjust based on where you extracted)
model_path = "..\models\imdb-distilbert"

# Load model and tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

# Set model to eval mode
model.eval()

# Example inference function
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probs).item()
        label = "Positive 😊" if predicted_class == 1 else "Negative 😞"
        return label, probs.squeeze().tolist()

# Example usage
if __name__ == "__main__":
    review = input("Enter a movie review: ")
    sentiment, scores = predict_sentiment(review)
    print(f"\nPredicted Sentiment: {sentiment}")
    print(f"Confidence Scores: {scores}")
