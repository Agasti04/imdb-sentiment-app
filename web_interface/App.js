async function classifySentiment() {
  const text = document.getElementById('review').value;
  const resultDiv = document.getElementById('result');

  if (!text.trim()) {
    resultDiv.innerText = "⚠️ Please enter a review.";
    return;
  }

  resultDiv.innerText = "⏳ Predicting...";

  try {
    const response = await fetch('http://localhost:8000/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ text })
    });

    const data = await response.json();

    resultDiv.innerText = `✅ Sentiment: ${data.label}\n\nConfidence: Negative ${data.confidence[0].toFixed(4)}, Positive ${data.confidence[1].toFixed(4)}`;
  } catch (err) {
    resultDiv.innerText = "❌ Error: Unable to reach the backend server.";
  }
}
