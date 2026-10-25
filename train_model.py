
import pandas as pd
import re
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load dataset
data = pd.read_csv("dataset/news.csv")

# Step 2: Remove rows with missing text or labels
data = data.dropna(subset=["title", "text", "label"])

# Step 3: Combine title and article text
data["content"] = data["title"] + " " + data["text"]

# Step 4: Clean the text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

data["content"] = data["content"].apply(clean_text)

# Step 5: Separate input and output
X = data["content"]
y = data["label"]

# Step 6: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training articles:", len(X_train))
print("Testing articles:", len(X_test))

# Step 7: Convert text into TF-IDF features
vectorizer = TfidfVectorizer(
    max_features=50000,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF training matrix:", X_train_tfidf.shape)

# Step 8: Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Step 9: Make predictions
y_pred = model.predict(X_test_tfidf)

# Step 10: Evaluate model
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Fake", "Real"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Step 11: Save the trained model and vectorizer

os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/fake_news_model.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

print("\nModel and vectorizer saved successfully!")

