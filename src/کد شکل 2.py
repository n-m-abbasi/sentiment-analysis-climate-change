import pandas as pd
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
import umap
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("twitter_sentiment_data_updated.csv")
df_sample = df.sample(n=3000, random_state=42).copy()

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Zآ-ی\s]", "", text)
    return text.lower().strip()

df_sample["clean_text"] = df_sample["message"].apply(clean_text)

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(df_sample["clean_text"].tolist(), show_progress_bar=True)

dbscan = DBSCAN(eps=0.5, min_samples=10, metric="cosine")
df_sample["cluster"] = dbscan.fit_predict(embeddings)

reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(embeddings)

plt.figure(figsize=(8,6))
palette = sns.color_palette("hls", len(set(df_sample["cluster"])))
sns.scatterplot(x=X_umap[:,0], y=X_umap[:,1], hue=df_sample["cluster"], palette=palette, s=30)
plt.title("Figure 2: UMAP Scatter Plot of Tweet Clusters (BERT + DBSCAN)", fontsize=14)
plt.xlabel("UMAP Component 1")
plt.ylabel("UMAP Component 2")
plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()