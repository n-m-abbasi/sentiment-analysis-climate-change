import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import numpy as np

df = pd.read_csv("twitter_sentiment_data_updated.csv")
df = df.sample(3000, random_state=42)  

def clean_text(text):
    text = re.sub(r"http\S+", "", text)     
    text = re.sub(r"@\w+", "", text)        
    text = re.sub(r"#", "", text)           
    text = re.sub(r"[^a-zA-Z\s]", "", text)  
    return text.lower().strip()

df["clean_text"] = df["message"].apply(clean_text)

vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
X = vectorizer.fit_transform(df["clean_text"])

for n_clusters in [2, 4, 5]:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df[f"cluster_{n_clusters}"] = kmeans.fit_predict(X)

    tsne = TSNE(n_components=2, random_state=42, perplexity=20, max_iter=500)
    X_tsne = tsne.fit_transform(X.toarray())

    plt.figure(figsize=(8,6))
    sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=df[f"cluster_{n_clusters}"], palette="Set2", s=30)
    plt.title(f"t-SNE Scatter Plot with {n_clusters} Clusters")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend(title="Cluster")
    plt.tight_layout()
    plt.show()

    print(f"\nTop keywords per cluster (k={n_clusters}):")
    tfidf_array = X.toarray()
    cluster_labels = df[f"cluster_{n_clusters}"]
    for cluster_id in range(n_clusters):
        indices = np.where(cluster_labels == cluster_id)[0]
        cluster_tfidf = tfidf_array[indices]
        mean_tfidf = np.mean(cluster_tfidf, axis=0)
        top_indices = mean_tfidf.argsort()[::-1][:10]
        top_keywords = [vectorizer.get_feature_names_out()[i] for i in top_indices]
        print(f"Cluster {cluster_id}: {', '.join(top_keywords)}")
