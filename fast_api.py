from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import re
import nltk
import uvicorn
import os
import requests

from nltk.corpus import stopwords

# Download stopwords if not already
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# === Download models if not available ===
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

FILES = {
    "sentiment_random_forest_model.pkl": "https://drive.google.com/uc?id=1VHVjlCQeT3lqjKD4a3DCvVC89EUg6F3e",
    "tfidf_vectorizer.pkl": "https://drive.google.com/uc?id=1XufPqd7h3kY3QKQt3mXbDTsrLOqOChbA",
    "label_encoder.pkl": "https://drive.google.com/uc?id=Y1RGQUfCU6NJH6SDbvF34ECITdOBFBiAJO",
}


def download_file(name, url):
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        print(f"Downloading {name}...")
        response = requests.get(url)
        with open(path, "wb") as f:
            f.write(response.content)
        print(f"{name} downloaded.")
    return path


# Download and load models
model = joblib.load(
    download_file(
        "sentiment_random_forest_model.pkl", FILES["sentiment_random_forest_model.pkl"]
    )
)
vectorizer = joblib.load(
    download_file("tfidf_vectorizer.pkl", FILES["tfidf_vectorizer.pkl"])
)
label_encoder = joblib.load(
    download_file("label_encoder.pkl", FILES["label_encoder.pkl"])
)


# === Text Preprocessing ===
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


# === FastAPI App ===
class TweetRequest(BaseModel):
    text: str


app = FastAPI()


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


# Local testing
if __name__ == "__main__":
    uvicorn.run("fast_api:app", host="0.0.0.0", port=8000, reload=True)


# uvicorn fast_api:app --reload
