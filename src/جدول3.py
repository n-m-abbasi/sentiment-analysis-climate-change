import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from textblob import TextBlob

df = pd.read_csv("twitter_sentiment_data_updated.csv")

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower().strip()

df["clean_text"] = df["message"].apply(clean_text)

vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
X = vectorizer.fit_transform(df["clean_text"])

kmeans = KMeans(n_clusters=3, random_state=42)
df["cluster"] = kmeans.fit_predict(X)

df["score"] = df["message"].apply(lambda x: TextBlob(x).sentiment.polarity)

table3 = df.groupby("cluster").agg(
    tweet_count=("message", "count"),
    mean_sentiment=("score", "mean")
).reset_index()

print(table3)
