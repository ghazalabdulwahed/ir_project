# -*- coding: utf-8 -*-
"""
Project: SOA Search Engine - Academic Project
Module: Semantic Embedding & Retrieval Service (BERT)
Author: Student 2 - AI Module
Date: June 2026
Description: Production-ready semantic search service using
Sentence-BERT embeddings + cosine similarity search.
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time

# =========================================================
# LOAD DATA
# =========================================================
def load_data(path="processed_documents.pkl"):
    df = pd.read_pickle(path)
    print(f"Dataset loaded: {df.shape}")
    return df


# =========================================================
# LOAD MODEL
# =========================================================
def load_model():
    print("Loading BERT model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("Model loaded successfully")
    return model


# =========================================================
# BUILD EMBEDDINGS
# =========================================================
def build_embeddings(model, texts):
    print("Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    print("Embeddings shape:", embeddings.shape)
    return embeddings


# =========================================================
# SAVE / LOAD EMBEDDINGS
# =========================================================
def save_embeddings(embeddings, path="medical_embeddings.npy"):
    np.save(path, embeddings)
    print("Embeddings saved to:", path)


def load_embeddings(path="medical_embeddings.npy"):
    return np.load(path)


# =========================================================
# SEMANTIC SEARCH
# =========================================================
def semantic_search(query, model, embeddings, texts, top_k=5):
    query_vec = model.encode([query])
    scores = cosine_similarity(query_vec, embeddings)[0]

    top_idx = scores.argsort()[::-1][:top_k]

    results = []
    for idx in top_idx:
        results.append({
            "text": texts[idx],
            "score": float(scores[idx])
        })

    return results


# =========================================================
# MAIN PIPELINE
# =========================================================
def main():
    start = time.time()

    # Load dataset
    df = load_data("processed_documents.pkl")
    texts = df["cleaned_text"].astype(str).tolist()

    # Use sample first (optional safety)
    # texts = texts[:10000]

    # Load model
    model = load_model()

    # Build embeddings
    embeddings = build_embeddings(model, texts)

    # Save embeddings
    save_embeddings(embeddings)

    # TEST QUERY
    query = "diabetes treatment"
    print("\nQuery:", query)

    results = semantic_search(query, model, embeddings, texts)

    print("\nTop Results:")
    for r in results:
        print("Score:", round(r["score"], 4))
        print("Text:", r["text"][:200])
        print("-" * 50)

    print("\nDone in:", round(time.time() - start, 2), "seconds")


if __name__ == "__main__":
    main()