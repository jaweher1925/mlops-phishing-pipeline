from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import re
import os

app = FastAPI(title="Phishing Email Classifier", version="1.0")

# Load champion model and vectorizer
with open("models/champion_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("models/champion_name.txt", "r") as f:
    champion_name = f.read().strip()

# Label mapping
LABELS = {0: "Legitimate", 1: "Phishing"}

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

class EmailRequest(BaseModel):
    subject: str = ""
    body: str = ""

class PredictionResponse(BaseModel):
    prediction: int
    label: str
    confidence: float
    model_used: str

@app.get("/")
def root():
    return {"message": "Phishing Email Classifier API", "model": champion_name}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": champion_name}

@app.post("/predict", response_model=PredictionResponse)
def predict(email: EmailRequest):
    text = email.subject + " " + email.body
    text = clean_text(text)
    vec = vectorizer.transform([text])
    prediction = int(model.predict(vec)[0])
    proba = model.predict_proba(vec)[0]
    confidence = float(max(proba))

    return PredictionResponse(
        prediction=prediction,
        label=LABELS.get(prediction, "Unknown"),
        confidence=round(confidence, 4),
        model_used=champion_name
    )