import os
import re
import joblib
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -----------------------------
# Page Configuration (ONLY config here)
# -----------------------------
st.set_page_config(
    page_title="SENTIMENT ANALYSER",
    page_icon="💬",
    layout="centered"
)

# -----------------------------
# Custom Dark Glass Theme (CSS)
# -----------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: #eaeaea;
        font-family: 'Inter', sans-serif;
    }

    textarea {
        background-color: #111827 !important;
        color: #e5e7eb !important;
        border-radius: 12px !important;
        border: 1px solid #374151 !important;
    }

    button[kind="primary"] {
        background: linear-gradient(90deg, #7c3aed, #9333ea);
        color: white !important;
        border-radius: 12px !important;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        transition: 0.3s;
    }

    button[kind="primary"]:hover {
        transform: scale(1.03);
        background: linear-gradient(90deg, #9333ea, #7c3aed);
    }

    div[data-testid="metric-container"] {
        background-color: rgba(17, 24, 39, 0.8);
        border-radius: 14px;
        padding: 1rem;
        border: 1px solid #374151;
    }

    .stProgress > div > div {
        background-image: linear-gradient(90deg, #22c55e, #16a34a);
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
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
