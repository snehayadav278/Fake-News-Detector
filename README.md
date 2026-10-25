# Fake News Detector

A machine learning web application that predicts whether a news article resembles fake or real news using Natural Language Processing (NLP).

## Technologies Used
- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- Streamlit

## Features
- Accepts news headlines and articles as input.
- Preprocesses text using NLP techniques.
- Predicts likely fake or real news.
- Provides a simple interactive web interface.

## Model Performance
Achieved approximately **98.81% accuracy** on the test split of the training dataset.

## Dataset
Used the Kaggle Fake and Real News Dataset containing fake and real news articles.

## How to Run
1. Install the required libraries:
   ```bash
   pip install pandas scikit-learn streamlit joblib
   ```
2. Run the application:
   ```bash
   streamlit run app.py
   ```

## Limitation
This model identifies text patterns and does not independently verify facts. Its test accuracy does not guarantee reliable predictions for recent news.

## Author
Sneha 
