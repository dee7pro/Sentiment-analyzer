import os
import re
import joblib
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="SENTIMENT ANALYSER",
    page_icon="💬",
    layout="centered"
)

# -----------------------------
# Paths (Streamlit-safe)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------
# Load NLTK resources (cached)
# -----------------------------
@st.cache_resource
def setup_nltk():
    nltk.download("stopwords")
    nltk.download("wordnet")

setup_nltk()

# -----------------------------
# Load model & vectorizer ONCE
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(BASE_DIR, "sentiment_model.pkl"))
    vectorizer = joblib.load(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"))
    return model, vectorizer

model, vectorizer = load_model()

# -----------------------------
# Text preprocessing
# -----------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = " ".join(w for w in text.split() if w not in stop_words)
    text = " ".join(lemmatizer.lemmatize(w) for w in text.split())
    return text

# -----------------------------
# UI
# -----------------------------
st.title("SENTIMENT ANALYSER")
st.caption("AI-powered sentiment analytics for product reviews")

review = st.text_area(
    "Paste a product review below",
    height=160,
    placeholder="Example: The shuttlecock quality is excellent and lasts long..."
)

if st.button("Analyze Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review to analyze.")
    else:
        cleaned = preprocess(review)
        vector = vectorizer.transform([cleaned])

        probabilities = model.predict_proba(vector)[0]
        prediction = probabilities.argmax()
        confidence = round(probabilities[prediction] * 100, 1)

        if prediction == 1:
            st.success("✅ Positive Sentiment")
        else:
            st.error("❌ Negative Sentiment")

        st.metric("Confidence Score", f"{confidence}%")
        st.progress(confidence / 100)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Built by Deepika A | Deployed on Streamlit Cloud")
