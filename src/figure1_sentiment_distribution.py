import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("twitter_sentiment_data.csv")

analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    vader_score = analyzer.polarity_scores(text)["compound"]
    score = (polarity + vader_score) / 2
    if score > 0.5:
        label = "Very Positive"
    elif score > 0.1:
        label = "Positive"
    elif score < -0.5:
        label = "Very Negative"
    elif score < -0.1:
        label = "Negative"
    else:
        label = "Neutral"
    return pd.Series([score, label])


df[["score", "sentiment_label"]] = df["message"].apply(analyze_sentiment)

print(df["sentiment_label"].value_counts())

sns.set(style="whitegrid", palette="pastel")
plt.figure(figsize=(8,6))
sns.countplot(data=df, x="sentiment_label", order=df["sentiment_label"].value_counts().index)
plt.title("Estimated Sentiment Distribution (TextBlob + VADER)")
plt.xlabel("Sentiment Label")
plt.ylabel("Tweet Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
