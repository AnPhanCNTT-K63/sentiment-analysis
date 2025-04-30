from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import re
import nltk
import uvicorn

from nltk.corpus import stopwords

# Download stopwords if not already
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Load exported model and preprocessing tools
model = joblib.load("models/sentiment_random_forest_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")


# Preprocessing function (same as training)
def preprocess_text(text):
    text = text.lower()

    emoji_pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f"
        "\U0001f300-\U0001f5ff"
        "\U0001f680-\U0001f6ff"
        "\U0001f1e0-\U0001f1ff"
        "\U00002500-\U00002bef"
        "\U00002702-\U000027b0"
        "\U000024c2-\U0001f251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2b55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"
        "\u3030"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub(r"", text)

    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\S+", "", text)
    text = re.sub(r"<unk>", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    text = " ".join([word for word in text.split() if word not in stop_words])
    return text


# Request format
class TweetRequest(BaseModel):
    text: str


# Create FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
def predict_sentiment(request: TweetRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    clean_text = preprocess_text(request.text)
    vectorized = vectorizer.transform([clean_text])
    prediction = model.predict(vectorized)[0]
    sentiment_label = label_encoder.inverse_transform([prediction])[0]

    return {
        "text": request.text,
        "clean_text": clean_text,
        "predicted_sentiment": sentiment_label,
    }


# For local testing
if __name__ == "__main__":
    uvicorn.run("fast_api:app", host="0.0.0.0", port=8000, reload=True)


# uvicorn fast_api:app --reload
