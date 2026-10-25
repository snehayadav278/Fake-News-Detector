
import streamlit as st
import joblib
import re

# Load the saved model and vectorizer
model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")


# Clean the input text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# App heading
st.title("📰 Fake News Detector")
st.write(
    "Enter a news article below to predict "
    "whether it resembles fake or real news "
    "in the training dataset."
)

# News input
news_text = st.text_area(
    "Paste your news article here:",
    height=250,
    placeholder="Enter the news headline and article text..."
)

# Prediction button
if st.button("Predict News", type="primary"):

    if not news_text.strip():
        st.warning("Please enter a news article first.")

    else:
        # Clean the input
        cleaned_text = clean_text(news_text)

        # Convert text into TF-IDF features
        text_features = vectorizer.transform([cleaned_text])

        # Predict the label
        prediction = model.predict(text_features)[0]

        # Display the result
        if prediction == 0:
            st.error("Prediction: Likely Fake News")
        else:
            st.success("Prediction: Likely Real News")

        st.caption(
            "This is an ML prediction, not a fact-check. "
            "Verify important claims using reliable sources."
        )
