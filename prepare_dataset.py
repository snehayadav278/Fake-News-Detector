
import pandas as pd

# Load both datasets
fake_news = pd.read_csv("dataset/Fake.csv")
real_news = pd.read_csv("dataset/True.csv")

# Assign labels
fake_news["label"] = 0
real_news["label"] = 1

# Combine both datasets
data = pd.concat([fake_news, real_news], ignore_index=True)

# Keep only the columns we need
data = data[["title", "text", "label"]]

# Shuffle the rows
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Display the results
print("Combined dataset:")
print(data.head())

print("\nTotal articles:", len(data))
print("\nLabel counts:")
print(data["label"].value_counts())

# Save the combined dataset
data.to_csv("dataset/news.csv", index=False)

print("\nDataset saved successfully!")
