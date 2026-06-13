import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util

class SemanticSearchService:
    def __init__(self, 
                 model_name='sentence-transformers/all-MiniLM-L6-v2', 
                 csv_path='quora_cleaned_sample.csv', 
                 embeddings_path='quora_embeddings.npy'):
        """
        تهيئة الخدمة وتحديد مسارات الملفات والنموذج المستخدم.
        """
        self.model_name = model_name
        self.csv_path = csv_path
        self.embeddings_path = embeddings_path
        self.model = None
        self.df = None
        self.all_embeddings = None

    def initialize_service(self, load_embeddings=True):
        """
        تحميل النموذج والبيانات، وتحميل الأشعة المحفوظة مسبقاً إذا كانت موجودة.
        """
        print("====== [Initialization] ======")
        print(f"Loading Model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        print("Model Loaded Successfully.")

        print(f"Reading Data from: {self.csv_path}...")
        if os.path.exists(self.csv_path):
            self.df = pd.read_csv(self.csv_path)
            # تنظيف أسطر النصوص المفقودة لضمان عدم تمرير قيم فارغة للنموذج
            self.df = self.df.dropna(subset=["cleaned_text"])
            print(f"Data Loaded Successfully. Shape: {self.df.shape}")
        else:
            raise FileNotFoundError(f"Error: {self.csv_path} not found!")

        if load_embeddings:
            if os.path.exists(self.embeddings_path):
                print(f"Loading Pre-computed Embeddings from: {self.embeddings_path}...")
                self.all_embeddings = np.load(self.embeddings_path)
                print(f"Embeddings Loaded. Shape: {self.all_embeddings.shape}")
            else:
                print(f"Warning: {self.embeddings_path} not found. You need to compute embeddings first.")

    def compute_and_save_all_embeddings(self, batch_size=32):
        """
        تحويل كافة المستندات في ملف الـ CSV إلى أشعة وحفظها محلياً (مهمة اليوم 3).
        """
        print("\n====== [Embedding Generation] ======")
        if self.df is None:
            raise ValueError("Service not initialized. Call initialize_service(load_embeddings=False) first.")
        
        all_docs = self.df["cleaned_text"].astype(str).tolist()
        print(f"Encoding {len(all_docs)} documents... (Batch Size: {batch_size})")
        
        self.all_embeddings = self.model.encode(
            all_docs,
            batch_size=batch_size,
            show_progress_bar=True
        )
        
        np.save(self.embeddings_path, self.all_embeddings)
        print(f"All embeddings saved successfully to '{self.embeddings_path}'")

    def search_semantic(self, query_text, top_k=5):
        """
        تنفيذ عملية البحث الدلالي وحساب جيب التمام لإرجاع أفضل النتائج المتطابقة (مهمة اليوم 2).
        """
        if self.all_embeddings is None:
            raise ValueError("Embeddings are not loaded or computed yet.")
            
        # 1. تحويل استعلام المستخدم إلى شعاع
        query_embedding = self.model.encode([query_text])
        
        # 2. حساب مصفوفة جيب التمام (Cosine Similarity) باستخدام محرك المكتبة الأصلي والسريع
        scores = util.cos_sim(query_embedding, self.all_embeddings)[0]
        
        # 3. ترتيب النتائج تنازلياً وجلب أعلى k عناصر
        top_indices = scores.argsort(descending=True)[:top_k]
        
        results = []
        for idx in top_indices:
            idx_item = int(idx)
            results.append({
               "doc_id": self.df.iloc[idx_item]["doc_id"],
        "score": round(float(scores[idx_item]), 4),
        "text": self.df.iloc[idx_item]["cleaned_text"]
            })
        return results


# الكود بالأسفل يعمل فقط عند تشغيل هذا الملف مباشرة كسكربت مستقل
if __name__ == "__main__":
    # إنشاء غرض من الخدمة
    search_service = SemanticSearchService()
    
    # تهيئة الخدمة بدون تحميل الأشعة القديمة لأننا سنقوم ببنائها الآن
    search_service.initialize_service(load_embeddings=False)
    
    # 1. تشغيل وتوليد وحفظ أشعة الـ 10,000 وثيقة (تنفذ مرة واحدة لتجهيز النظام)
    search_service.compute_and_save_all_embeddings(batch_size=32)
    
    # إعادة تحميل الخدمة الآن للتأكد من أن عملية الحفظ والقراءة من الملف تعمل 100%
    search_service.initialize_service(load_embeddings=True)
    
    # 2. تجربة الفحص والبحث (فحص دالة التشابه والاسترجاع)
    test_query = "invest in stock market"
    print(f"\n====== [Testing Search Query]: '{test_query}' ======")
    
    search_results = search_service.search_semantic(test_query, top_k=5)
    
    for i, res in enumerate(search_results, 1):
        print(f"Rank {i} | Similarity Score: {res['score']}")
        print(f"Document [{res['index']}]: {res['text']}")
        print("-" * 60)