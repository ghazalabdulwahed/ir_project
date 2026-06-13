import numpy as np
import pandas as pd
import os

# =====================================================================
# 1. استدعاء الخدمات من كافة الملفات
# =====================================================================
from retrieval_service import build_inverted_index, prepare_bm25, bm25_search
from embedding_service import SemanticSearchService
from lama_evaluation_rag_service import EvaluationService, RAGService

print("⚙️ جاري تجهيز النظام بالكامل...")

# بناء الفهرس لـ BM25
inverted_index = build_inverted_index(df)
bm25_idf, avg_doc_length = prepare_bm25(df, inverted_index)

# تهيئة خدمة الـ BERT (بدون تحميل أشعة قديمة لأننا سنحسب مباشرة للوثائق المسترجعة)
bert_service = SemanticSearchService()
bert_service.initialize_service(load_embeddings=False)

# تهيئة خدمات التقييم والـ RAG
eval_service = EvaluationService()
rag_service = RAGService()

print("✅ تم تحميل وتجهيز كافة الخدمات بنجاح!")


# =====================================================================
# 2. بناء تابع البحث التسلسلي المشترك (Cascade Search Function)
# =====================================================================
def cascade_search_pipeline(query, top_n_bm25=50, top_k_final=5):
    """
    المرحلة الأولى: جلب أعلى N وثيقة باستخدام BM25 السريع.
    المرحلة الثانية: إعادة ترتيب هذه الـ N وثيقة فقط باستخدام نموذج BERT.
    """
    # 1. البحث باستخدام BM25
    bm25_results = bm25_search(query, df, inverted_index, bm25_idf, avg_doc_length)
    top_bm25_candidates = bm25_results[:top_n_bm25] # أخذ أعلى 50 وثيقة مثلاً
    
    if not top_bm25_candidates:
        return []
        
    # استخراج المعرفات والنصوص للوثائق المرشحة فقط
    candidate_ids = [doc_id for doc_id, _ in top_bm25_candidates]
    candidate_df = df[df["doc_id"].isin(candidate_ids)].copy()
    candidate_texts = candidate_df["cleaned_text"].astype(str).tolist()
    
    # 2. إعادة الترتيب بواسطة BERT
    query_embedding = bert_service.model.encode([query])
    doc_embeddings = bert_service.model.encode(candidate_texts, show_progress_bar=False)
    
    # حساب التشابه لجيب التمام (Cosine Similarity)
    from sentence_transformers import util
    scores = util.cos_sim(query_embedding, doc_embeddings)[0].tolist()
    
    # ربط المعرفات بالسكور الجديد والترتيب تنازلياً
    ranked_results = sorted(
        zip(candidate_df["doc_id"].tolist(), candidate_texts, scores),
        key=lambda x: x[2],
        reverse=True
    )
    
    return ranked_results[:top_k_final]


# =====================================================================
# 3. تشغيل السيناريو الكامل (End-to-End Demo)
# =====================================================================
print("\n🚀 --- تشغيل محاكاة البحث والـ RAG والتقييم ---")

# استعلام تجريبي
user_query = "how to learn machine learning"
print(f"🔍 استعلام المستخدم: '{user_query}'")

# تشغيل الـ Pipeline المشترك
final_results = cascade_search_pipeline(user_query, top_n_bm25=20, top_k_final=5)

# عرض النتائج المسترجعة النهائية
print("\n🔝 أفضل 3 وثائق مسترجعة بعد إعادة الترتيب بـ BERT:")
retrieved_ids_for_eval = []
retrieved_texts_for_rag = []

for rank, (doc_id, text, score) in enumerate(final_results[:3], 1):
    print(f"المرتبة {rank} | معرف الوثيقة: {doc_id} | السكور الدلالي: {round(score, 4)}")
    print(f"📄 النص: {text}\n" + "-"*40)
    retrieved_ids_for_eval.append(doc_id)
    retrieved_texts_for_rag.append(text)

# تنفيذ ميزة الـ RAG الذكية بناءً على هذه النتائج
rag_answer = rag_service.generate_smart_answer(user_query, retrieved_texts_for_rag)
print(f"\n🤖 إجابة نظام الـ RAG التوليدي:\n{rag_answer}")


# =====================================================================
# 4. تقييم جودة النظام (Evaluation)
# =====================================================================
print("\n📊 --- تقييم جودة الاسترجاع ---")

# سنفترض جدلاً أن هذه هي المعرفات الحقيقية المطلوبة للاختبار (Ground Truth)
mock_relevant_ids = [final_results[0][0], final_results[1][0], 999] 

# حساب المقاييس
metrics = eval_service.evaluate_model(
    all_retrieved=[[doc_id for doc_id, _, _ in final_results]], 
    all_relevant=[mock_relevant_ids], 
    k=5
)

print("📝 المقاييس المحسوبة للنظام التسلسلي:")
for metric_name, value in metrics.items():
    print(f"🔹 {metric_name}: {value}")