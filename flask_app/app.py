from flask import Flask, render_template, request
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Flask app
flask_app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# NLTK setup
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = " ".join([w for w in text.split() if w not in stop_words])
    text = " ".join([lemmatizer.lemmatize(w) for w in text.split()])
    return text

@flask_app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    sentiment = None

    if request.method == "POST":
        review = request.form["review"]
        processed = preprocess(review)
        vector = vectorizer.transform([processed])

        proba = model.predict_proba(vector)[0]
        confidence = round(max(proba) * 100, 2)
        pred = model.predict(vector)[0]

        if pred == 1:
            prediction = "Positive 😊"
            sentiment = "positive"
        else:
            prediction = "Negative 😞"
            sentiment = "negative"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        sentiment=sentiment
    )

from flask import Flask, render_template, request
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Flask app
flask_app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# NLTK setup
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = " ".join([w for w in text.split() if w not in stop_words])
    text = " ".join([lemmatizer.lemmatize(w) for w in text.split()])
    return text

@flask_app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    sentiment = None

    if request.method == "POST":
        review = request.form["review"]
        processed = preprocess(review)
        vector = vectorizer.transform([processed])

        proba = model.predict_proba(vector)[0]
        confidence = round(max(proba) * 100, 2)
        pred = model.predict(vector)[0]

        if pred == 1:
            prediction = "Positive 😊"
            sentiment = "positive"
        else:
            prediction = "Negative 😞"
            sentiment = "negative"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        sentiment=sentiment
    )


