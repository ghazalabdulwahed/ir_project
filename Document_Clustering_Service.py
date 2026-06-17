# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_pickle("processed_documents.pkl")
print("Dataset shape:", df.shape)

# =========================================================
# LOAD EMBEDDINGS
# =========================================================
embeddings = np.load("medical_embeddings.npy")
print("Embeddings shape:", embeddings.shape)

# =========================================================
# KMEANS CLUSTERING
# =========================================================
k = 10

kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(embeddings)

print("First 20 clusters:", clusters[:20])

# =========================================================
# ADD CLUSTER TO DATAFRAME
# =========================================================
df["cluster"] = clusters

print("\nSample Data:")
print(df.head())

# =========================================================
# SHOW CLUSTERS
# =========================================================
for i in range(k):
    print(f"\n===== Cluster {i} =====")

    sample = df[df["cluster"] == i]["cleaned_text"].head(5)

    for text in sample:
        print("-", text)

# =========================================================
# SAVE OUTPUT
# =========================================================
df.to_pickle("clustered_documents.pkl")
print("\nSaved successfully!")

# =========================================================
# VERIFY
# =========================================================
df2 = pd.read_pickle("clustered_documents.pkl")

print("\nFinal Shape:", df2.shape)
print("\nCluster Distribution:")
print(df2["cluster"].value_counts())