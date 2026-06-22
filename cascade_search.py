# -*- coding: utf-8 -*-
"""
Project: SOA Search Engine - Academic Project
Module: Serial Cascade Hybrid Search Service (BM25 Filtering -> BERT Re-ranking)
Author: Maria Alskal
Date: June 2026
Description: Two-stage retrieval. Stage 1 utilizes fast library Rank-BM25 to filter 
             top candidate documents. Stage 2 applies dense BERT embeddings and 
             vectorized Cosine Similarity to rerank candidates.
"""

import time
import pickle
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# استدعاء دالة المعالجة المحدثة من خدمتك الأساسية
from search_core_service import preprocess_text

# استدعاء دالات الـ BERT من خدمة الـ Embedding الخاصة بزميلتك
from BERT_Embedding_Service import (
    load_model,
    load_embeddings
)

# ==================================================
# 1. Load Processed Dataset & Mapping Metadata
# ==================================================
print("⏳ Loading processed documents and metadata...")
df = pd.read_pickle("processed_documents.pkl")
doc_ids = df["doc_id"].tolist()
print(f"✔ Dataset Loaded: {len(doc_ids):,} documents found.")

# إنشاء قاموس الفهرسة السريعة لربط معرّف الوثيقة بموقعها في المصفوفة O(1)
docid_to_row = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
print("✔ Doc ID → Matrix Row Mapping Ready.")

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
# 3. Load BERT Model and Dense Embeddings
# ==================================================
print("⏳ Loading BERT model and semantic embeddings...")
model = load_model()
embeddings = load_embeddings("medical_embeddings.npy")
print("✔ BERT Semantic Models Ready.")


# ==================================================
# 4. VECTORIZED SERIAL CASCADE SEARCH ENGINE
# ==================================================
def serial_cascade_search(query, top_bm25=500, final_k=10):
    """
    Stage 1: Fast keyword filtering via Rank-BM25 library.
    Stage 2: High-speed vectorized BERT Re-ranking using Matrix operations.
    """
    # 1. معالجة وتجهيز الكلمات للاستعلام
    processed_query = preprocess_text(query)
    tokenized_query = processed_query.split()

    if not tokenized_query:
        print("⚠️ Query terms are completely filtered. Returning empty results.")
        return []

    # --- المرحلة الأولى (Stage 1): فلترة سريعة بـ BM25 ---
    start_stage1 = time.time()
    doc_scores = bm25_model.get_scores(tokenized_query)
    raw_results = list(zip(doc_ids, doc_scores))
    
    # جلب أفضل وثائق مرشحة (مثلاً أفضل 500 وثيقة)
    candidate_docs = sorted(raw_results, key=lambda x: x[1], reverse=True)[:top_bm25]
    
    # فلترةCandidate docs لتجنب إدخال وثائق سكورها صفر بالخطأ
    candidate_docs = [doc for doc in candidate_docs if doc[1] > 0]
    
    print(f"✔ [Stage 1] BM25 Filtered {len(candidate_docs)} candidates in {(time.time() - start_stage1)*1000:.2f} ms.")

    if not candidate_docs:
        return []

    # --- المرحلة الثانية (Stage 2): إعادة الترتيب الدلالي بـ BERT ---
    start_stage2 = time.time()
    
    # تحويل الاستعلام الحالي إلى شعاع BERT دلالي
    query_embedding = model.encode([query]).reshape(1, -1)

    # جلب مصفوفة الأشعة للوثائق المرشحة فقط بضربة واحدة وبشكل متوازٍ (Vectorized Slice)
    candidate_row_indices = [docid_to_row[doc_id] for doc_id, _ in candidate_docs]
    candidate_embeddings = embeddings[candidate_row_indices]  # مصفوفة بحجم (500, 768) مثلاً

    # حساب تشابه جيب التمام (Cosine Similarity) للـ 500 وثيقة دفعة واحدة خارج الحلقات والتكرارات
    similarity_matrix = cosine_similarity(query_embedding, candidate_embeddings)[0]

    # ربط معرفات الوثائق بالسكور الجديد بعد الفرز الدلالي
    reranked_results = []
    for idx, (doc_id, _) in enumerate(candidate_docs):
        reranked_results.append((doc_id, similarity_matrix[idx]))

    # إعادة الترتيب التنازلي بناءً على سكور BERT الجديد
    reranked_results.sort(key=lambda x: x[1], reverse=True)
    
    print(f"✔ [Stage 2] BERT Semantic Re-ranking Complete in {(time.time() - start_stage2)*1000:.2f} ms.")
    
    # إرجاع أفضل الـ K النهائيات المطلوبة لعرضهم بالواجهة (مثلاً أفضل 10)
    return reranked_results[:final_k]


# ==================================================
# Test Serial Cascade Search Execution
# ==================================================
if __name__ == "__main__":

    query = "diabetes treatment"
    print(f"\n🚀 EXECUTING SERIAL CASCADE SEARCH FOR QUERY: '{query}'")
    print("=" * 60)

    start_total = time.time()
    cascade_results = serial_cascade_search(query=query, top_bm25=500, final_k=10)
    total_duration = (time.time() - start_total) * 1000
    
    print(f"✔ Total Cascade Executed in {total_duration:.2f} ms.")
    print("=" * 60)

    # عرض النتائج الـ 10 الأولى بالتفصيل مع نصوصها
    if not cascade_results:
        print("No matching results found.")
    else:
        print(f"\n🔝 TOP {len(cascade_results)} RE-RANKED CASCADE RESULTS:")
        
        # بناء قاموس الفحص السريع للنصوص من أجل سرعة الطباعة O(1)
        fast_text_lookup = dict(zip(df["doc_id"], df["cleaned_text"]))
        
        for rank, (doc_id, score) in enumerate(cascade_results, start=1):
            print(f"\nRank {rank:02d} | Doc ID: {doc_id}")
            print(f"Refined Semantic Score (BERT Cosine): {score:.4f}")
            
            # طباعة مقتطف صغير لمعاينة جودة المحرك التسلسلي الهجين
            preview_text = fast_text_lookup.get(doc_id, "No text available.")[:250]
            print(f"Document Preview:\n   {preview_text}...")
            print("-" * 60)