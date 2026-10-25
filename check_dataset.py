
import pandas as pd

fake_news = pd.read_csv("dataset/Fake.csv")
real_news = pd.read_csv("dataset/True.csv")

print("Fake News Dataset:")
print(fake_news.head())

print("\nReal News Dataset:")
print(real_news.head())

print("\nFake news rows:", len(fake_news))
print("Real news rows:", len(real_news))

print("\nFake news columns:", fake_news.columns.tolist())
print("Real news columns:", real_news.columns.tolist())
