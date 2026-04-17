import gradio as gr
import pickle
import re

with open("models/champion_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("models/champion_name.txt", "r") as f:
    champion_name = f.read().strip()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict(subject, body):
    text = clean_text(subject + " " + body)
    vec = vectorizer.transform([text])
    prediction = int(model.predict(vec)[0])
    proba = model.predict_proba(vec)[0]
    confidence = round(float(max(proba)) * 100, 2)
    if prediction == 1:
        return f"Phishing Email Detected\nConfidence: {confidence}%\nModel: {champion_name}"
    else:
        return f"Legitimate Email\nConfidence: {confidence}%\nModel: {champion_name}"

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Email Subject", placeholder="e.g. Urgent: Verify your account"),
        gr.Textbox(label="Email Body", lines=5, placeholder="Paste email body here..."),
    ],
    outputs=gr.Textbox(label="Prediction Result"),
    title="Phishing Email Classifier",
    description="MLOps Project — Jaweher Hichri ",
    examples=[
        ["Urgent: Verify your account", "Click here to verify your bank account or it will be suspended"],
        ["Meeting tomorrow at 3pm", "Hi team, reminder we have a meeting tomorrow at 3pm in room 204"],
        ["You won a prize!", "Congratulations! Click the link to claim your reward now."],
    ]
)

demo.launch()