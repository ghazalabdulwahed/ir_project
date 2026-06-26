# -*- coding: utf-8 -*-
"""
Project: SOA Search Engine
Module: Vector Store & Indexing Service (FAISS)
Description: Service to build and query a high-speed FAISS vector index.
"""

import numpy as np
import faiss
import time
import os

def build_faiss_index(embeddings_path="medical_embeddings.npy", index_path="faiss_index.bin"):
    print(f"⏳ جاري تحميل الأشعة من {embeddings_path}...")
    # FAISS يتعامل حصراً مع مصفوفات float32
    embeddings = np.load(embeddings_path).astype('float32')
    
    dimension = embeddings.shape[1]
    
    # توحيد الأشعة (Normalization) لنتمكن من حساب Cosine Similarity عبر Inner Product
    faiss.normalize_L2(embeddings)
    
    # بناء الفهرس
    print("🧠 جاري بناء فهرس FAISS للبحث السريع...")
    start_time = time.time()
    index = faiss.IndexFlatIP(dimension) # IP = Inner Product
    index.add(embeddings)
    
    # حفظ الفهرس
    faiss.write_index(index, index_path)
    print(f"✅ تم بناء وحفظ الفهرس بنجاح في {time.time() - start_time:.2f} ثانية!")
    return index

def load_faiss_index(index_path="faiss_index.bin"):
    """دالة لتحميل الفهرس الجاهز للاستخدام في واجهة المستخدم"""
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"❌ الفهرس {index_path} غير موجود. الرجاء بناؤه أولاً.")
    return faiss.read_index(index_path)

def search_faiss_index(query_vector, index, top_k=10):
    """دالة للبحث السريع داخل الفهرس"""
    # التأكد من نوع البيانات وتوحيد شعاع الاستعلام
    query_vector = query_vector.astype('float32')
    faiss.normalize_L2(query_vector)
    
    # البحث
    distances, indices = index.search(query_vector, top_k)
    return distances[0], indices[0]

if __name__ == "__main__":
    # تشغيل هذا السكربت مرة واحدة فقط لبناء الفهرس
    build_faiss_index()