import os
import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "sentiment_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"))



# Page config
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="💬",
    layout="centered"
)

# Download NLTK data (Streamlit-safe)
@st.cache_resource
def load_nltk():
    nltk.download("stopwords")
    nltk.download("wordnet")

load_nltk()

# Load model & vectorizer
@st.cache_resource
def load_model():
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = " ".join(w for w in text.split() if w not in stop_words)
    text = " ".join(lemmatizer.lemmatize(w) for w in text.split())
    return text

# ---------- UI ----------
st.title("SENTIMENT ANALYZER")
st.caption("AI-powered sentiment analysis of product reviews")

review = st.text_area(
    "Paste a product review below",
    height=150,
    placeholder="Paste a product review here..."
)

if st.button("Analyze sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        processed = preprocess(review)
        vector = vectorizer.transform([processed])

        proba = model.predict_proba(vector)[0]
        pred = proba.argmax()
        confidence = round(proba[pred] * 100, 1)

        if pred == 1:
            st.success("Positive 😊")
        else:
            st.error("Negative 😞")

        st.metric("Confidence", f"{confidence}%")
        st.progress(confidence / 100)

st.markdown("---")
st.caption("Built by Deepika A")
