# -*- coding: utf-8 -*-
"""
Project: SOA Search Engine - Academic Project
Module: Document Indexing & Retrieval Service (TF-IDF & BM25 using industry-standard libraries)
Author: Maria Alskal (Student 1 - Data & Core IR Models)
Date: June 2026
Description: Production-ready Python service using Scikit-Learn for TF-IDF 
             and Rank-BM25 for dynamic document retrieval.
"""

import time
import pickle
import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# مكاتب سوق العمل القياسية والمطلوبة
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

# =====================================================================
# SYSTEM INITIALIZATION & TEXT PREPROCESSING
# =====================================================================
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    """Clean, tokenize, and stem text for standard representation."""
    tokens = word_tokenize(str(text).lower())
    cleaned_tokens = [
        stemmer.stem(word)
        for word in tokens
        if word.isalnum() and word not in stop_words
    ]
    return " ".join(cleaned_tokens)


# =====================================================================
# 📅 INDUSTRY-STANDARD MODELS (TF-IDF & BM25)
# =====================================================================

def train_tfidf_vectorizer(corpus):
    """Trains Scikit-Learn's TF-IDF Vectorizer and transforms the corpus."""
    print("⏳ [SKLEARN] Generating TF-IDF vectors using Scikit-Learn...")
    start_time = time.time()
    
    # استخدام مكتبة sklearn لحساب TF-IDF وبناء المصفوفة المتباعدة (Sparse Matrix)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    print(f"✔ TF-IDF Matrix shape: {tfidf_matrix.shape}")
    print(f"✔ TF-IDF computation complete in {time.time() - start_time:.2f} seconds.")
    return vectorizer, tfidf_matrix


def train_bm25_model(corpus):
    """Initializes the Rank-BM25 model using tokenized text corpus."""
    print("⏳ [RANK-BM25] Initializing BM25Okapi model from library...")
    start_time = time.time()
    
    # مكتبة rank_bm25 تحتاج الداتا كـ قائمة من الكلمات المتوضعة داخل قائمة
    tokenized_corpus = [doc.split() for doc in corpus]
    bm25_model = BM25Okapi(tokenized_corpus)
    
    print(f"✔ BM25 model initialized in {time.time() - start_time:.2f} seconds.")
    return bm25_model


# =====================================================================
# SERVICE PIPELINE EXECUTION & ASSET EXPORT
# =====================================================================

def main():
    print("==================================================")
    print("🚀 RUNNING CORE SEARCH SERVICE PIPELINE (LIBRARIES VERSION)")
    print("==================================================")
    
    # أسماء ملفات الكاش لحفظ النماذج الجاهزة
    tfidf_file = "tfidf_assets.pkl"
    bm25_file = "bm25_model.pkl"
    meta_file = "metadata.pkl"
    
    # الفحص الذكي للملفات المحفوظة لمنع إعادة الحسابات المستهلكة للوقت
    if os.path.exists(tfidf_file) and os.path.exists(bm25_file) and os.path.exists(meta_file):
        print("⚡ [O(1) Speed] Found precomputed library assets! Loading files instantly...")
        start_load = time.time()
        
        with open(tfidf_file, "rb") as f:
            vectorizer, tfidf_matrix = pickle.load(f)
        with open(bm25_file, "rb") as f:
            bm25_model = pickle.load(f)
        with open(meta_file, "rb") as f:
            doc_ids = pickle.load(f)
            
        print(f"✅ Library assets loaded successfully in {time.time() - start_load:.4f} seconds! (Zero Lag)")
        
    else:
        print("⚠️ Precomputed assets not found. Running full industry pipeline...")
        
        # تحميل الداتا الأساسية المجهزة مسبقاً
        df = pd.read_pickle("processed_documents.pkl")
        corpus = df["cleaned_text"].astype(str).tolist()
        doc_ids = df["doc_id"].tolist()
        
        # 1. حساب الـ TF-IDF عبر مكتبة Sklearn
        vectorizer, tfidf_matrix = train_tfidf_vectorizer(corpus)
        
        # 2. حساب الـ BM25 عبر مكتبة Rank-BM25
        bm25_model = train_bm25_model(corpus)
        
        # حفظ المكاتب والنماذج المدربة لضمان عدم تكرار البناء عند تشغيل الواجهة
        print("💾 Saving trained models for future instant loads...")
        with open(tfidf_file, "wb") as f: 
            pickle.dump((vectorizer, tfidf_matrix), f)
        with open(bm25_file, "wb") as f: 
            pickle.dump(bm25_model, f)
        with open(meta_file, "wb") as f: 
            pickle.dump(doc_ids, f)
        print("✅ Models cached successfully.")

    # 🔍 تجربة محرك البحث الاحترافي للتأكد من السلامة والأداء
    print("\n--------------------------------------------------")
    print("🔍 RUNNING LIVE SERVICE VERIFICATION (RANK-BM25)")
    print("--------------------------------------------------")
    
    test_query = "calcium channel"
    
    # 1. معالجة الاستعلام الأساسية
    processed_query = preprocess_text(test_query)
    tokenized_query = processed_query.split()
    
    print(f"Original Query: '{test_query}' -> Processed: '{processed_query}'")
    
    # 🛡️ الفحص الذكي: حماية النظام من الاستعلامات الفارغة أو التي تحتوي على Stopwords فقط
    if not tokenized_query:
        print("⚠️ Warning: Query contains no valid terms after preprocessing. Returning 0 results.")
        active_results = []
    else:
        start_search = time.time()
        
        # حساب سكور الـ BM25 للاستعلام الحالي لكل الوثائق دفعة واحدة بكفاءة عالية
        doc_scores = bm25_model.get_scores(tokenized_query)
        
        # ربط السكور بالـ doc_id وترتيب النتائج تنازلياً
        results = list(zip(doc_ids, doc_scores))
        results = sorted(results, key=lambda x: x[1], reverse=True)
        
        # فلترة النتائج ذات الصلة (التي تملك سكور أكبر من صفر)
        active_results = [r for r in results if r[1] > 0]
        search_duration = (time.time() - start_search) * 1000
        print(f"✔ Search completed in {search_duration:.2f} ms. Found {len(active_results):,} matching docs.")
    
    # عرض أعلى 5 نتائج للتأكد من صحة الحسابات والأرقام
    print("\n🔝 Top 5 Results:")
    if active_results:
        for rank, (d_id, score) in enumerate(active_results[:5], 1):
            print(f"   Rank {rank}: Doc ID = {d_id} | BM25 Score = {score:.4f}")
    else:
        print("   No matching documents found.")

if __name__ == "__main__":
    main()