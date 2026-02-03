🧠 Sentilitics — AI Sentiment Analyzer

Sentilitics is an end-to-end machine learning–powered sentiment analysis web application that classifies product reviews as Positive or Negative and displays a confidence score in real time.

The project demonstrates the complete lifecycle of an ML solution — from data preprocessing and feature engineering to model training, evaluation, and cloud deployment using Streamlit.

Deployed on Streamlit Community Cloud : https://sentilitics.streamlit.app/

📌 Problem Statement

Online product reviews contain valuable insights about customer satisfaction.
Manually analyzing thousands of reviews is time-consuming and error-prone.

Objective:
Build an automated system that:

Classifies customer reviews into Positive or Negative

Shows prediction confidence

Works in real time via a web interface

🗂 Dataset

Source: Flipkart product reviews (scraped by data engineering team)

Product: YONEX MAVIS 350 Nylon Shuttle

Records: 8,518 reviews

Key Features:

Review text

Ratings

Review Title

Review Date

Upvotes / Downvotes

🧹 Exploratory Data Analysis (EDA)

Performed in Jupyter Notebook:

Checked and handled missing values

Removed duplicate reviews

Analyzed:

Review length distribution

Rating vs sentiment relationship

Class imbalance

Identified noise and outliers in text length


🔧 Data Preprocessing

Lowercasing text

Removing punctuation and special characters

Stopword removal (NLTK)

Lemmatization using WordNet

Cleaned text used for feature extraction

🔢 Feature Engineering

Implemented and compared:

Bag of Words (BoW)

TF-IDF Vectorization ✅ (Best performing)

(Word2Vec & BERT discussed conceptually)

🤖 Model Training & Evaluation

Trained multiple machine learning models:

Model	F1-Score
Logistic Regression	0.92
Linear SVM	0.92
Naive Bayes	0.91

📌 Metric Used: F1-Score
Chosen due to class imbalance and to balance precision & recall.

Final Model: Logistic Regression + TF-IDF

🌐 Web Application

Built using Streamlit with:

Text input for reviews

Real-time sentiment prediction

Confidence meter with progress bar

☁️ Deployment

Hosted on Streamlit Community Cloud

Model artifacts loaded efficiently using caching

Portable and cloud-safe file handling

Deepika A
Data Science & Machine Learning Enthusiast
