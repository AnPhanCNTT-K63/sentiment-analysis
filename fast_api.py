from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
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


# Custom response middleware (like TransformInterceptor)
@app.middleware("http")
async def add_custom_response_structure(request: Request, call_next):
    try:
        response = await call_next(request)
        body = await response.body()
        return JSONResponse(
            content={
                "data": response.json() if hasattr(response, "json") else body.decode(),
                "message": "OK",
                "statusCode": response.status_code,
            },
            status_code=response.status_code,
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "data": None,
                "message": str(e),
                "statusCode": 500,
            },
        )


# Create FastAPI app
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


# For local testing
if __name__ == "__main__":
    uvicorn.run("fast_api:app", host="0.0.0.0", port=8000, reload=True)


# uvicorn fast_api:app --reload
