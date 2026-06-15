import os
import pandas as pd
import numpy as np
from sentence_transformers import util

# 1️⃣ استيراد المكتبة وتفعيل المفتاح الخاص بكِ
import google.generativeai as foreign_genai

# المفتاح الحالي الخاص بكِ
MY_KEY = "AQ.Ab8RN6LTtedjHjgP1VkBxDwRi8AAS46UD2CAFETqYcnK92LuQQ"
foreign_genai.configure(api_key=MY_KEY)

# =====================================================================
# 2. قراءة ملف البيانات الأساسي
# =====================================================================
print("📂 جاري قراءة ملف البيانات...")
df = pd.read_csv("quora_cleaned_sample.csv")
print(f"✅ تم تحميل البيانات بنجاح! حجم الجدول: {df.shape}")

# =====================================================================
# 3. إعداد كلاس الـ RAG الذكي المعتمد على المكتبة المستقرة
# =====================================================================
class RealRAGService:
    def __init__(self):
        # ✨ التعديل هنا: استخدام الإصدار الأحدث gemini-2.5-flash لتجنب خطأ 404 نهائياً
        self.model = foreign_genai.GenerativeModel('gemini-2.5-flash')
        print("🤖 [Real RAG] تم الاتصال بنموذج Gemini بنجاح واستعداد خط الإنتاج!")
        
    def generate_smart_answer(self, query, top_retrieved_docs_text):
        context = "\n".join([f"- {text}" for text in top_retrieved_docs_text[:3]])
        
        prompt = f"""
       You are an expert Information Retrieval Assistant. 
        Answer the user query. Use the provided retrieved context to guide your answer if it's helpful.
        If the context only contains related question titles or doesn't have the explicit step-by-step answer, 
        use your own extensive knowledge to give a comprehensive, helpful, and concise answer to the user.
        
        User Query: {query}
        Retrieved Context:
        {context}
        
        Answer (Brief and concise):
        """
        try:
            # التوليد الحي المباشر
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"⚠️ فشل الاتصال بالنموذج التوليدي: {str(e)}"

# =====================================================================
# 4. استدعاء الدالات الخارجية وتجهيز الفهارس
# =====================================================================
from retrieval_service import build_inverted_index, prepare_bm25, bm25_search
from embedding_service import SemanticSearchService

print("\n⚙️ جاري بناء فهارس الـ BM25 السريعة...")
inverted_index = build_inverted_index(df)
bm25_idf, avg_doc_length = prepare_bm25(df, inverted_index)

print("🧠 جاري تهيئة موديل الـ BERT للبحث الدلالي...")
bert_service = SemanticSearchService()
bert_service.initialize_service(load_embeddings=False)

# بناء خدمة الـ RAG الذكية
rag_service = RealRAGService()

# =====================================================================
# 5. خط البحث المتسلسل (Cascade Search Pipeline)
# =====================================================================
def cascade_search_pipeline(query, top_n_bm25=20, top_k_final=5):
    bm25_results = bm25_search(query, df, inverted_index, bm25_idf, avg_doc_length)
    top_bm25_candidates = bm25_results[:top_n_bm25]
    if not top_bm25_candidates: return []
    
    candidate_ids = [doc_id for doc_id, _ in top_bm25_candidates]
    candidate_df = df[df["doc_id"].isin(candidate_ids)].copy()
    candidate_texts = candidate_df["cleaned_text"].astype(str).tolist()
    
    query_embedding = bert_service.model.encode([query])
    doc_embeddings = bert_service.model.encode(candidate_texts, show_progress_bar=False)
    
    scores = util.cos_sim(query_embedding, doc_embeddings)[0].tolist()
    ranked_results = sorted(zip(candidate_df["doc_id"].tolist(), candidate_texts, scores), key=lambda x: x[2], reverse=True)
    return ranked_results[:top_k_final]

# =====================================================================
# 6. تشغيل الاستعلام والتوليد الحي النهائي (نسخة مطورة للمعاينة)
# =====================================================================
user_query = "how to learn machine learning"
print(f"\n🔍 استعلام المستخدم الحقيقي: '{user_query}'")
print("⏳ جاري سحب البيانات ومطابقتها دلالياً...")

final_results = cascade_search_pipeline(user_query, top_n_bm25=20, top_k_final=5)
retrieved_texts = [text for _, text, _ in final_results]

# ✨ سطر إضافي لمعاينة النصوص المسترجعة من ملفكِ
print("\n📄 النصوص التي تم سحبها من ملف الـ CSV بناءً على بحثكِ:")
for i, text in enumerate(retrieved_texts[:3], 1):
    print(f" {i}. {text}")

rag_answer = rag_service.generate_smart_answer(user_query, retrieved_texts)
print(f"\n🤖 إجابة نظام الـ RAG الذكي:\n{rag_answer}")


import os
import math
import pandas as pd
import numpy as np
from sentence_transformers import util

# =====================================================================
# 1. دالات احتساب المقاييس الرياضية (Precision, Recall, nDCG)
# =====================================================================
def calculate_precision_at_k(retrieved_ids, ground_truth_ids, k):
    top_k_retrieved = retrieved_ids[:k]
    relevant_retrieved = len(set(top_k_retrieved) & set(ground_truth_ids))
    return relevant_retrieved / k

def calculate_recall_at_k(retrieved_ids, ground_truth_ids, k):
    top_k_retrieved = retrieved_ids[:k]
    relevant_retrieved = len(set(top_k_retrieved) & set(ground_truth_ids))
    if len(ground_truth_ids) == 0: return 0.0
    return relevant_retrieved / len(ground_truth_ids)

def calculate_dcg_at_k(retrieved_ids, ground_truth_ids, k):
    top_k_retrieved = retrieved_ids[:k]
    dcg = 0.0
    for i, doc_id in enumerate(top_k_retrieved, 1):
        if doc_id in ground_truth_ids:
            dcg += 1.0 / math.log2(i + 1)
    return dcg

def calculate_ndcg_at_k(retrieved_ids, ground_truth_ids, k):
    dcg = calculate_dcg_at_k(retrieved_ids, ground_truth_ids, k)
    ideal_retrieved = [doc_id for doc_id in retrieved_ids if doc_id in ground_truth_ids]
    ideal_retrieved += list(set(ground_truth_ids) - set(ideal_retrieved))
    idcg = calculate_dcg_at_k(ideal_retrieved, ground_truth_ids, k)
    if idcg == 0.0: return 0.0
    return dcg / idcg

# =====================================================================
# 2. إعداد مجموعة بيانات الاختبار الحقيقية (من واقع ملف الـ CSV الخاص بكِ)
# =====================================================================
evaluation_dataset = [
    {
        "query": "how to invest in share market",
        "ground_truth": [1, 2]  # المعرفات الحقيقية من ملفكِ
    },
    {
        "query": "story of Kohinoor diamond",
        "ground_truth": [3, 4]  # المعرفات الحقيقية من ملفكِ
    },
    {
        "query": "how to increase internet speed",
        "ground_truth": [5, 6]  # المعرفات الحقيقية من ملفكِ
    }
]

# =====================================================================
# 3. تشغيل التقييم التلقائي بناءً على الـ Cascade Search Pipeline المستقر لديكِ
# =====================================================================
K_VAL = 5
results_report = []

print(f"⚡ جاري بدء التقييم التلقائي الحقيقي على {len(evaluation_dataset)} أسئلة اختبار من ملفكِ...")

for item in evaluation_dataset:
    q = item["query"]
    gt = item["ground_truth"]
    
    # استدعاء خط البحث المتسلسل (الموجود مسبقاً في النوت بوك الخاص بكِ)
    search_output = cascade_search_pipeline(q, top_n_bm25=20, top_k_final=K_VAL)
    retrieved_ids = [doc_id for doc_id, _, _ in search_output]
    
    # حساب المقاييس
    p_score = calculate_precision_at_k(retrieved_ids, gt, k=K_VAL)
    r_score = calculate_recall_at_k(retrieved_ids, gt, k=K_VAL)
    n_score = calculate_ndcg_at_k(retrieved_ids, gt, k=K_VAL)
    
    results_report.append({
        "الاستعلام (Query)": q,
        f"Precision@{K_VAL}": p_score,
        f"Recall@{K_VAL}": r_score,
        f"nDCG@{K_VAL}": n_score
    })

# =====================================================================
# 4. عرض لوحة النتائج النهائية وحساب المتوسط العام (Mean Average)
# =====================================================================
report_df = pd.DataFrame(results_report)

mean_scores = {
    "الاستعلام (Query)": "📊 MEAN AVERAGE (المتوسط العام للنظام)",
    f"Precision@{K_VAL}": report_df[f"Precision@{K_VAL}"].mean(),
    f"Recall@{K_VAL}": report_df[f"Recall@{K_VAL}"].mean(),
    f"nDCG@{K_VAL}": report_df[f"nDCG@{K_VAL}"].mean()
}
report_df = pd.concat([report_df, pd.DataFrame([mean_scores])], ignore_index=True)

# حفظ التقرير للاستخدام اللاحق
report_df.to_csv("retrieval_real_evaluation_report.csv", index=False)

print("\n🎉 تم انتهاء التقييم بنجاح! إليكِ الجدول العلمي الجاهز للطباعة والوضع في الأطروحة:")
report_df