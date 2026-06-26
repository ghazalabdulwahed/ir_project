# -*- coding: utf-8 -*-
"""
Project: SOA Search Engine - Academic Project
Module: Production Learning-to-Rank (LTR) Service
Author: Academic Group Team (Optimized SOA Pipeline)
Date: June 2026
Description: High-performance LTR execution pipeline strictly utilizing precomputed 
             BM25 scores, precalculated static SBERT embeddings, and a single-instance 
             BERT model loaded via shared SOA services.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sentence_transformers import util

# استيراد الخدمات الأصلية المجهزة من الزملاء (دون إعادة إنشاء النماذج)
from search_core_service import preprocess_text
from BERT_Embedding_Service import load_model, load_embeddings

# ==========================================
# 1. تهيئة النماذج والبيانات الأساسية (مرة واحدة)
# ==========================================
print("📁 جاري تحميل البيانات المهيأة (Pickle)...")
df = pd.read_pickle("processed_documents.pkl")
df = df.reset_index(drop=True)

df["doc_id"] = df["doc_id"].astype(str)
df["cleaned_text"] = df["cleaned_text"].astype(str)

doc_ids = df["doc_id"].tolist()
texts = df["cleaned_text"].tolist()

# تحميل الخدمات المشتركة لعدم تكرار التحميل أو الاستهلاك
print("🧠 جاري تحميل المودل ومصفوفة تمثيلات BERT الدلالية (SOA Architecture)...")
shared_bert_model = load_model()                     # تحميل المودل (مرة واحدة)
embeddings = load_embeddings("medical_embeddings.npy") # تحميل الـ Embeddings (مرة واحدة)

# تحميل نموذج BM25 المحفوظ مسبقاً
with open("bm25_model.pkl", "rb") as f:
    bm25_model = pickle.load(f)

# ==========================================
# 2. استخراج الخصائص (Feature Extraction)
# ==========================================
def extract_ltr_features(query_emb, doc_idx, bm25_score):
    """
    استخراج الخصائص لزوج (استعلام - مستند) بأداء فائق O(1).
    يعتمد على الترميز المحفوظ مسبقاً للمستند والترميز الممرر للاستعلام.
    """
    doc_emb = embeddings[doc_idx]
    
    # حساب التشابه الدلالي (Cosine Similarity)
    bert_score = util.cos_sim(query_emb, doc_emb).item()
    doc_length = len(texts[doc_idx].split())

    return [
        bm25_score,
        bert_score,
        doc_length
    ]

# ==========================================
# 3. التدريب أو الاسترجاع النموذجي (LTR Persistency)
# ==========================================
ltr_model_file = "ltr_model.pkl"

if os.path.exists(ltr_model_file):
    print("💾 جاري تحميل نموذج الترتيب LTR المحفوظ مسبقاً...")
    with open(ltr_model_file, "rb") as f:
        ltr_ranker = pickle.load(f)
else:
    print("📝 نموذج LTR غير موجود. جاري تحضير مصفوفة التدريب والتدريب لأول مرة...")
    
    queries = []
    relevant_docs = {}
    
    # ملاحظة: إذا كان لديك ملف استعلامات حقيقي (tsv/csv)، يُستبدل هذا الجزء بـ pd.read_csv("queries.csv")
    for i in range(min(100, len(df))):
        query = " ".join(df.iloc[i]["cleaned_text"].split()[:5])
        doc_id = df.iloc[i]["doc_id"]
        queries.append(query)
        relevant_docs[query] = [doc_id]

    X, y = [], []
    counter = 0

    for query in queries:
        # ترميز الاستعلام مرة واحدة فقط لكل Query باستخدام الخدمة المشتركة
        q_emb = shared_bert_model.encode(query, convert_to_tensor=True)
        
        tokenized_query = preprocess_text(query).split()
        scores = bm25_model.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:20]

        for idx in top_indices:
            counter += 1
            doc_id = df.iloc[idx]["doc_id"]

            features = extract_ltr_features(q_emb, idx, scores[idx])
            X.append(features)
            y.append(1 if doc_id in relevant_docs.get(query, []) else 0)

    X = np.array(X)
    y = np.array(y)

    print(f"⚖️ تدريب نموذج التصنيف Logistic Regression (بأوزان متوازنة)...")
    ltr_ranker = LogisticRegression(max_iter=2000, class_weight="balanced")
    ltr_ranker.fit(X, y)
    
    # حفظ المودل لعدم تكرار التدريب مستقبلاً
    with open(ltr_model_file, "wb") as f:
        pickle.dump(ltr_ranker, f)
    print("✅ اكتمل التدريب وتم حفظ نموذج LTR بنجاح!")

# ==========================================
# 4. الترتيب النهائي (Re-ranking Service)
# ==========================================
def ltr_ranked_search(query_example, top_k=10):
    """
    إعادة ترتيب النتائج (Re-rank) بالاعتماد على الخدمات الثابتة.
    يتم ترميز الاستعلام مرة واحدة فقط خارج الحلقة باستخدام الخدمة المشتركة.
    """
    print(f"\n📌 جاري استرجاع وإعادة ترتيب النتائج للاستعلام: '{query_example}'")
    
    processed_q = preprocess_text(query_example)
    tokenized_query = processed_q.split()
    
    if not tokenized_query:
        return []

    # 1. ترميز الاستعلام مرة واحدة فقط بأداء عالٍ
    query_emb = shared_bert_model.encode(query_example, convert_to_tensor=True)

    # 2. استرجاع أفضل 20 وثيقة ترشيح مبدئي من BM25
    scores = bm25_model.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:20]

    results = []
    for idx in top_indices:
        doc_id = df.iloc[idx]["doc_id"]

        # 3. استخراج الخصائص بالاعتماد على الـ Embeddings الجاهزة (فهرس الوثيقة index)
        features = extract_ltr_features(query_emb, idx, scores[idx])
        
        # الحصول على احتمالية أن المستند ذو صلة (احتمال الفئة 1)
        score = ltr_ranker.predict_proba([features])[0][1]
        results.append((doc_id, score))

    # الترتيب التنازلي بناءً على السكور (الاحتمالية) بعد الدمج
    results = sorted(results, key=lambda x: x[1], reverse=True)
    return results[:top_k]

# ==========================================
# 5. التنفيذ التجريبي لخدمة الاسترجاع
# ==========================================
if __name__ == "__main__":
    test_results = ltr_ranked_search("lead burden neuropsychology", top_k=10)
    
    print("أفضل 10 مستندات تم ترتيبها:")
    for rank, (doc_id, sc) in enumerate(test_results, 1):
        print(f"Rank {rank:02d} | Doc ID: {doc_id} | LTR Score: {sc:.4f}")