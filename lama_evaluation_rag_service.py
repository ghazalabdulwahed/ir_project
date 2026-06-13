import numpy as np
import pandas as pd

# =====================================================================
# اليوم 1: التوابع والمقاييس الأساسية (Precision & Recall)
# =====================================================================

def precision_at_k(retrieved_ids, relevant_ids, k=10):
    # حساب الدقة عند الترتيب K (Precision@K)
    # يقيس نسبة الوثائق ذات الصلة المسترجعة في أول K نتيجة مقارنة بالعدد الكلي للنتائج المسترجعة.
    top_k_retrieved = retrieved_ids[:k]
    if not top_k_retrieved:
        return 0.0
    
    true_positives = len(set(top_k_retrieved) & set(relevant_ids))
    return true_positives / len(top_k_retrieved)


def calculate_recall(retrieved_ids, relevant_ids):
    # حساب الاستدعاء الكامل (Recall)
    # يقيس نسبة الوثائق ذات الصلة التي استطاع النظام استرجاعها من بين جميع الوثائق ذات الصلة الحقيقية.
    if not relevant_ids:
        return 0.0
    
    true_positives = len(set(retrieved_ids) & set(relevant_ids))
    return true_positives / len(relevant_ids)


# =====================================================================
# اليوم 2: المقاييس الحساسة للترتيب (MAP & nDCG)
# =====================================================================

def calculate_average_precision(retrieved_ids, relevant_ids):
    # حساب متوسط الدقة (Average Precision - AP) لاستعلام واحد مع مراعاة الترتيب.
    if not relevant_ids:
        return 0.0
    
    ap_sum = 0.0
    true_positives_count = 0
    
    for rank, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            true_positives_count += 1
            precision_at_rank = true_positives_count / (rank + 1)
            ap_sum += precision_at_rank
            
    if true_positives_count == 0:
        return 0.0
        
    return ap_sum / len(relevant_ids)


def calculate_map(all_queries_retrieved, all_queries_relevant):
    # حساب الـ Mean Average Precision (MAP) لجميع الاستعلامات الاختبارية.
    ap_scores = [calculate_average_precision(ret, rel) for ret, rel in zip(all_queries_retrieved, all_queries_relevant)]
    return np.mean(ap_scores) if ap_scores else 0.0


def ndcg_at_k(retrieved_ids, relevant_ids, k=10):
    # حساب الـ Normalized Discounted Cumulative Gain (nDCG@K) مع افتراض صلة ثنائية (0 أو 1).
    top_k_retrieved = retrieved_ids[:k]
    dcg = 0.0
    
    for rank, doc_id in enumerate(top_k_retrieved):
        if doc_id in relevant_ids:
            dcg += 1.0 / np.log2(rank + 2)
            
    idcg = 0.0
    actual_relevant_in_top_k = min(len(relevant_ids), k)
    for rank in range(actual_relevant_in_top_k):
        idcg += 1.0 / np.log2(rank + 2)
        
    if idcg == 0.0:
        return 0.0
        
    return dcg / idcg


# =====================================================================
# اليوم 3: تجميع الخدمات كـ SOA (Clean Architecture) وميزة الـ RAG
# =====================================================================

class EvaluationService:
    # خدمة التقييم المستقلة والمسؤولة عن احتساب جودة محرك البحث لكل النماذج والمجموعات.
    def __init__(self):
        pass
        
    def evaluate_model(self, all_retrieved, all_relevant, k=10):
        # تقوم بحساب المقاييس الأربعة دفعة واحدة لنموذج معين وتخرجها كقاموس منظم.
        p_at_k_scores = [precision_at_k(ret, rel, k) for ret, rel in zip(all_retrieved, all_relevant)]
        recall_scores = [calculate_recall(ret, rel) for ret, rel in zip(all_retrieved, all_relevant)]
        ndcg_scores = [ndcg_at_k(ret, rel, k) for ret, rel in zip(all_retrieved, all_relevant)]
        map_score = calculate_map(all_retrieved, all_relevant)
        
        return {
            f"Precision@{k}": round(np.mean(p_at_k_scores), 4),
            "Recall": round(np.mean(recall_scores), 4),
            f"nDCG@{k}": round(np.mean(ndcg_scores), 4),
            "MAP": round(map_score, 4)
        }


class RAGService:
    # خدمة الـ RAG (Retrieval-Augmented Generation) - الميزة الإضافية للمشروع.
    def __init__(self):
        print("RAG Service Initialized Successfully.")
        
    def generate_smart_answer(self, query, top_retrieved_docs_text):
        # دمج الوثائق المسترجعة لتوليد إجابة ذكية ومباشرة للمستعلم.
        context = " \\n ".join(top_retrieved_docs_text[:3])
        prompt = f"Based on the following documents:\\n{context}\\n\\nAnswer the user query: {query}"
        
        # محاكاة توليد النص من نموذج LLM
        mock_llm_response = f"[RAG Answer] summary based on the retrieved data for your query: '{query}'."
        return mock_llm_response


# =====================================================================
# اليوم 4: الفحص الشامل للخدمات والتشغيل الذاتي (Demo Block)
# =====================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("=== تشغيل الفحص التجريبي لخدمة التقييم والـ RAG ===")
    print("=" * 50)
    
    eval_server = EvaluationService()
    rag_server = RAGService()

    # بيانات تجريبية لـ 3 استعلامات افتراضية
    all_retrieved_demo = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    all_relevant_demo = [[2, 10], [5, 6], [1]]

    # 1. تنفيذ التقييم وحساب المقاييس
    results = eval_server.evaluate_model(all_retrieved_demo, all_relevant_demo, k=10)
    print("\\n📊 === نتائج تقييم النظام التجريبية ===")
    print(results)

    # 2. تنفيذ نظام الـ RAG الذكي
    sample_docs = ["Python is an interpreted language.", "Information Retrieval is fun."]
    rag_response = rag_server.generate_smart_answer("What is Python?", sample_docs)
    print("\\n🤖 === نتيجة نظام الـ RAG ===")
    print(rag_response)
    print("=" * 50)