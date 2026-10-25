
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample news articles
news = [
    "Government announces a new policy",
    "Government announces election results",
    "Sports team wins the championship"
]

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Convert text into numerical features
features = vectorizer.fit_transform(news)

# Display vocabulary
print("Words learned by TF-IDF:")
print(vectorizer.get_feature_names_out())

# Display numerical matrix
print("\nTF-IDF matrix:")
print(features.toarray())

# Display matrix dimensions
print("\nMatrix shape:", features.shape)
