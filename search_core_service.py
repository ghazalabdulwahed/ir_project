# -*- coding: utf-8 -*-
"""
Project: SOA Search Engine - Academic Project
Module: Document Indexing & Retrieval Service (TF-IDF & BM25)
Author: Maria Alskal (Student 1 - Data & Core IR Models)
Date: June 2026
Description: Production-ready Python service for high-performance indexing, 
             VSM TF-IDF vector generation, and dynamic BM25 retrieval.
"""

import math
import time
import pickle
from collections import defaultdict, Counter
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# =====================================================================
# SYSTEM INITIALIZATION & TEXT PREPROCESSING
# =====================================================================
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess_query(text):
    """Clean, tokenize, and stem the user query to match dataset tokens."""
    tokens = word_tokenize(str(text).lower())
    cleaned_tokens = [
        stemmer.stem(word)
        for word in tokens
        if word.isalnum() and word not in stop_words
    ]
    return " ".join(cleaned_tokens)


# =====================================================================
# 📅 DAY 1: INVERTED INDEX ARCHITECTURE
# =====================================================================
def build_inverted_index_and_stats(df):
    """
    Builds the Inverted Index mapping terms to Document IDs.
    Uses an optimized single-pass loop over the dataframe arrays.
    """
    print("⏳ [Day 1] Building Inverted Index and extracting core stats...")
    start_time = time.time()
    
    inverted_index = defaultdict(set)
    doc_lengths = {}
    doc_term_freqs = {}
    
    for doc_id, cleaned_text in zip(df["doc_id"], df["cleaned_text"]):
        words = str(cleaned_text).split()
        
        # 1. Store document length for BM25 normalization
        doc_lengths[doc_id] = len(words)
        
        # 2. Compute local term frequencies for TF-IDF
        counts = Counter(words)
        doc_term_freqs[doc_id] = counts
        
        # 3. Populate Inverted Index
        for word in counts.keys():
            inverted_index[word].add(doc_id)
            
    print(f"✔ Day 1 Complete. Vocabulary Size: {len(inverted_index):,} words.")
    print(f"   Execution time: {time.time() - start_time:.2f} seconds.")
    return inverted_index, doc_lengths, doc_term_freqs


# =====================================================================
# 📅 DAY 2: VECTOR SPACE MODEL (VSM) & TF-IDF WEIGHTING
# =====================================================================
def compute_global_idf(df, inverted_index):
    """Computes dampened global IDF weights for all terms: log(1 + N/df)."""
    print("⏳ [Day 2] Calculating global IDF weights...")
    start_time = time.time()
    N = len(df)
    idf_weights = {}
    
    for word, doc_ids in inverted_index.items():
        df_t = len(doc_ids)
        idf_weights[word] = math.log(1 + (N / df_t))
        
    print(f"✔ IDF calculation complete in {time.time() - start_time:.2f} seconds.")
    return idf_weights

def compute_vsm_tfidf(doc_term_freqs, idf_weights):
    """Generates TF-IDF sparse vectors for all documents using log(1 + tf) * idf."""
    print("⏳ [Day 2] Generating TF-IDF sparse vectors...")
    start_time = time.time()
    tfidf_vectors = {}
    
    for doc_id, term_counts in doc_term_freqs.items():
        tfidf_vectors[doc_id] = {}
        for word, count in term_counts.items():
            tf_smoothed = math.log(1 + count)
            word_idf = idf_weights.get(word, 0)
            tfidf_vectors[doc_id][word] = tf_smoothed * word_idf
            
    print(f"✔ TF-IDF vectors generated in {time.time() - start_time:.2f} seconds.")
    return tfidf_vectors


# =====================================================================
# 📅 DAY 3: ADVANCED PROBABILISTIC MODEL (BM25) & SEARCH ENGINE
# =====================================================================
def prepare_bm25_weights(df, inverted_index):
    """Precomputes probabilistic BM25 IDF weights to ensure zero search lag."""
    print("⏳ [Day 3] Precomputing BM25 probabilistic IDF weights...")
    start_time = time.time()
    N = len(df)
    bm25_idf = {}
    
    for word, doc_ids in inverted_index.items():
        df_term = len(doc_ids)
        bm25_idf[word] = math.log(((N - df_term + 0.5) / (df_term + 0.5)) + 1)
        
    print(f"✔ BM25 IDF weights prepared in {time.time() - start_time:.2f} seconds.")
    return bm25_idf

def bm25_search_engine(
    query,
    inverted_index,
    bm25_idf,
    avg_doc_length,
    doc_lengths,
    doc_term_freqs,
    k1=1.5,
    b=0.75
):
    """
    Executes a high-speed BM25 search over candidate documents using hash-maps.
    Accepts dynamic k1 and b parameters from the API Gateway / UI.
    """
    # Tokenize the query terms directly
    query_terms = str(query).lower().split()
    if not query_terms:
        return []
        
    # Boolean Filtering: Get candidate documents containing at least one query term
    candidate_docs = set()
    for term in query_terms:
        if term in inverted_index:
            candidate_docs.update(inverted_index[term])
            
    # Scoring loop over filtered candidates only
    scores = {}
    for doc_id in candidate_docs:
        doc_len = doc_lengths[doc_id]
        term_freqs = doc_term_freqs[doc_id]
        score = 0
        
        for term in query_terms:
            if term not in term_freqs:
                continue
                
            tf = term_freqs[term]
            term_idf = bm25_idf.get(term, 0)
            
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_len / avg_doc_length))
            
            score += term_idf * (numerator / denominator)
            
        scores[doc_id] = score
        
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# =====================================================================
# SERVICE PIPELINE EXECUTION & ASSET EXPORT
# =====================================================================
def main():
    print("==================================================")
    print("🚀 RUNNING CORE SEARCH SERVICE PIPELINE")
    print("==================================================")
    
    # Load Main Dataset
    print("Loading data...")
    df = pd.read_pickle("processed_documents.pkl")
    print(f"Dataset Loaded: {len(df):,} documents found.")
    
    # Run Day 1 Tasks
    inverted_index, doc_lengths, doc_term_freqs = build_inverted_index_and_stats(df)
    
    # Run Day 2 Tasks
    idf_weights = compute_global_idf(df, inverted_index)
    tfidf_vectors = compute_vsm_tfidf(doc_term_freqs, idf_weights)
    
    # Run Day 3 Tasks
    bm25_idf = prepare_bm25_weights(df, inverted_index)
    avg_doc_length = sum(doc_lengths.values()) / len(df)
    print(f"Global Average Document Length: {round(avg_doc_length, 2)}")

    # Verification Test
    print("\n--------------------------------------------------")
    print("🔍 RUNNING LIVE SERVICE VERIFICATION")
    print("--------------------------------------------------")
    test_query = "calcium channel"
    print(f"Query: '{test_query}'")
    
    start_search = time.time()
    results = bm25_search_engine(
        query=test_query,
        inverted_index=inverted_index,
        bm25_idf=bm25_idf,
        avg_doc_length=avg_doc_length,
        doc_lengths=doc_lengths,
        doc_term_freqs=doc_term_freqs
    )
    duration = time.time() - start_search
    print(f"✔ Search completed in {duration * 1000:.2f} ms. Found {len(results):,} docs.")
    print("Top 3 results:")
    for rank, (doc_id, score) in enumerate(results[:3], 1):
        print(f"  Rank {rank} | Doc ID: {doc_id:<12} | Score: {score:.4f}")
    print("--------------------------------------------------\n")


if __name__ == "__main__":
    main()