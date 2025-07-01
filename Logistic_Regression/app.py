import joblib
import gradio as gr
import re

# Load model and vectorizer
model = joblib.load("logreg_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Preprocess function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Predict function
def predict_sentiment(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    return "😊 Positive" if pred == 1 else "😠 Negative"

# Gradio interface
iface = gr.Interface(fn=predict_sentiment,
                     inputs=gr.Textbox(lines=2, placeholder="Enter text here..."),
                     outputs="text",
                     title="Sentiment Analyzer (Logistic Regression)",
                     description="A simple sentiment classifier trained on Sentiment140.")

iface.launch()
