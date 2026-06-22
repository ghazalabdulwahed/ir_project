# -*- coding: utf-8 -*-
"""
Project: SOA Search Engine - Academic Project
Module: Parallel Hybrid Search Service (BM25 + BERT via Reciprocal Rank Fusion)
Author: Maria Alskal
Date: June 2026
Description: Combines lexical keyword scores from Rank-BM25 and dense semantic 
             scores from BERT using high-performance Reciprocal Rank Fusion (RRF).
"""

import time
import pickle
import os
import pandas as pd
from collections import defaultdict

# استدعاء الميزات المحدثة من خدمة الـ BM25 المبنية بالمكاتب
from search_core_service import preprocess_text

# استدعاء دالات الـ BERT من خدمة الـ Embedding الخاصة بزميلتك
from BERT_Embedding_Service import (
    load_model,
    load_embeddings,
    semantic_search
)

# ==================================================
# 1. Load Processed Dataset & Metadata
# ==================================================
print("⏳ Loading processed documents and metadata...")
df = pd.read_pickle("processed_documents.pkl")
doc_ids = df["doc_id"].tolist()
texts = df["cleaned_text"].astype(str).tolist()
print(f"✔ Dataset Loaded: {len(doc_ids):,} documents found.")

# ==================================================
# 2. Load Precomputed BM25 Model Assets (From Libraries)
# ==================================================
print("⏳ Loading precomputed Rank-BM25 library assets...")
bm25_file = "bm25_model.pkl"

if not os.path.exists(bm25_file):
    raise FileNotFoundError(
        "❌ BM25 asset file not found! Please run 'search_core_service.py' first to build and cache the models."
    )

with open(bm25_file, "rb") as f:
    bm25_model = pickle.load(f)
print("✔ Rank-BM25 Library Model Ready.")

# ==================================================
# 3. Load BERT Model & Embeddings (AI Service)
# ==================================================
print("⏳ Loading BERT model and semantic embeddings...")
model = load_model()
embeddings = load_embeddings("medical_embeddings.npy")
print("✔ BERT Semantic Search Service Ready.")


# ==================================================
# 4. HIGH-PERFORMANCE RECIPROCAL RANK FUSION (RRF)
# ==================================================
def reciprocal_rank_fusion(bm25_results, bert_results, k=60):
    """
    Blends two independent ranked lists using the RRF algorithm.
    Optimized to run in O(N) using fast dictionary lookups instead of DataFrame scans.
    """
    fused_scores = defaultdict(float)

    # 1. صهر رتب نتائج الـ BM25
    for rank, (doc_id, score) in enumerate(bm25_results, start=1):
        fused_scores[doc_id] += 1.0 / (k + rank)

    # 2. صهر رتب نتائج الـ BERT (باستخدام معرف الوثيقة مباشرة لسرعة فائقة)
    for rank, result in enumerate(bert_results, start=1):
        # التأكد من جلب الـ doc_id المرفق مع النتيجة الدلالية مباشرة
        doc_id = result.get("doc_id")
        if doc_id:
            fused_scores[doc_id] += 1.0 / (k + rank)

    # الترتيب التنازلي بناءً على السكور النهائي بعد الدمج
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)


# ==================================================
# Test Parallel Hybrid Search Execution
# ==================================================
if __name__ == "__main__":

    query = "diabetes treatment"
    print(f"\n🚀 EXECUTING PARALLEL HYBRID SEARCH FOR QUERY: '{query}'")
    print("=" * 60)

    # ----------------------------------------------
    # أ. معالجة وتمرير طلب البحث لـ BM25
    # ----------------------------------------------
    processed_query = preprocess_text(query)
    tokenized_query = processed_query.split()

    if not tokenized_query:
        print("⚠️ Query terms are completely filtered. Aborting search.")
        bm25_results = []
    else:
        start_bm25 = time.time()
        doc_scores = bm25_model.get_scores(tokenized_query)
        raw_bm25 = list(zip(doc_ids, doc_scores))
        # جلب أفضل 100 وثيقة مرشحة من الـ BM25 للصهر
        bm25_results = sorted(raw_bm25, key=lambda x: x[1], reverse=True)[:100]
        print(f"✔ BM25 Retrieval Complete in {(time.time() - start_bm25)*1000:.2f} ms.")

    # ----------------------------------------------
    # ب. تمرير طلب البحث لـ BERT Semantic Search
    # ----------------------------------------------
    start_bert = time.time()
    bert_results = semantic_search(
        query=query,
        model=model,
        embeddings=embeddings,
        texts=texts,
        doc_ids=doc_ids,
        top_k=100  # جلب أفضل 100 وثيقة دلالية
    )
    print(f"✔ BERT Retrieval Complete in {(time.time() - start_bert)*1000:.2f} ms.")

    # ----------------------------------------------
    # ج. تنفيذ الدمج الهجين عبر التابع الفائق RRF
    # ----------------------------------------------
    start_fusion = time.time()
    hybrid_results = reciprocal_rank_fusion(bm25_results, bert_results, k=60)
    print(f"✔ Reciprocal Rank Fusion (RRF) Complete in {(time.time() - start_fusion)*1000:.2f} ms.")
    print(f"✔ Total Fused Documents in Active Pool: {len(hybrid_results)} docs.")

    # ----------------------------------------------
    # د. عرض النتائج النهائية الـ 10 الأولى بالتفصيل
    # ----------------------------------------------
    print("\n🔝 TOP 10 HYBRID SEARCH RESULTS (FINAL RANKING):")
    print("=" * 60)

    # بناء قاموس سريع لجلب نصوص الوثائق بلمح البصر O(1) عند الطباعة
    fast_df_lookup = dict(zip(df["doc_id"], df["cleaned_text"]))

    for rank, (doc_id, rrf_score) in enumerate(hybrid_results[:10], start=1):
        print(f"Rank {rank:02d} | Doc ID: {doc_id} | RRF Combined Score: {rrf_score:.6f}")
        # عرض مقتطف من النص لمعاينة جودة المطابقة اللغوية والدلالية
        preview_text = fast_df_lookup.get(doc_id, "No text available.")[:150]
        print(f"        Snippet: {preview_text}...")
        print("-" * 60)