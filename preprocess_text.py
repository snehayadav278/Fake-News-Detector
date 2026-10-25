
import re

def clean_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Test the function
sample = "BREAKING!!! Visit https://example.com for NEWS."

cleaned = clean_text(sample)

print("Original text:", sample)
print("Cleaned text:", cleaned)
