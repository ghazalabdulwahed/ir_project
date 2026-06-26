# -*- coding: utf-8 -*-
"""
Project: Integrated Information Retrieval & Clustering System (SOA Architecture)
Module: Tightly Coupled API Gateway & Streamlit UI (Front-End)
Author: Academic Group Team 
Date: June 2026
Description: Production-ready Streamlit interface leveraging cached library models
             (Rank-BM25 + SBERT + LTR) for real-time high-speed search & clustering.
"""

import os
import math
import pickle
import numpy as np
import pandas as pd
import re
from collections import Counter, defaultdict
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.corpus import stopwords
from sallynnnew_service import process_query_service, suggest_query
from sentence_transformers import SentenceTransformer

# =====================================================================

@st.cache_resource
def download_nltk_resources():
    packages = ['punkt', 'stopwords', 'wordnet']
    for package in packages:
        try:
            nltk.download(package, quiet=True)
        except Exception:
            pass

download_nltk_resources()
try:
    stop_words = set(stopwords.words('english'))
except Exception:
    stop_words = set()

# =====================================================================

def process_query_service(query, expand=True):
    from textblob import TextBlob
    corrected_query = str(TextBlob(query).correct())
    query_clean = re.sub(r'[^a-z\s]', '', corrected_query.lower())
    words = query_clean.split()

    if not expand:
        return " ".join(words)

    medical_syns = {
        "heart": ["cardiac"],
        "disease": ["illness", "disorder"],
        "cancer": ["tumor", "carcinoma"],
        "diabetes": ["diabetic"],
        "treatment": ["therapy"],
        "vaccination": ["immunization"],
        "immune": ["immunity"],
        "clinical": ["medical"],
        "trial": ["study"]
    }

    expanded_words = list(words)
    for w in words:
        if w in medical_syns:
            for syn in medical_syns[w]:
                if syn not in expanded_words:
                    expanded_words.append(syn)

    return " ".join(expanded_words)

# ==========================================================

st.set_page_config(page_title="Information Retrieval System 2026", layout="wide", page_icon="🔍")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700&display=swap');
    html, body, .stTextInput, .stButton, [data-testid="stSidebar"], p, div, h1, h2, h3 {
        font-family: 'Tajawal', sans-serif !important;
        direction: rtl; text-align: right;
    }
    [data-testid="stSidebar"] { background-color: #f1f5f9 !important; border-left: 2px solid #cbd5e1; }
    .sidebar-header-custom { color: #1e3a8a; font-size: 20px; font-weight: bold; border-bottom: 3px solid #3b82f6; padding-bottom: 8px; margin-bottom: 20px; text-align: center; }
    .sidebar-section-card { background-color: #ffffff; padding: 14px; border-radius: 8px; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.04); border-right: 4px solid #3b82f6; }
    .section-label { color: #1e3a8a; font-size: 14px; font-weight: 700; margin-bottom: 8px; }
    .names-frame { border: 2px solid #4a90e2; border-radius: 12px; padding: 20px; background-color: #f0f7ff; margin-bottom: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .project-title { color: #1e3a8a; font-size: 26px; font-weight: bold; text-align: center; margin-bottom: 10px; }
    .team-grid { display: flex; justify-content: center; gap: 15px; flex-wrap: wrap; margin-top: 15px; }
    .member-badge { background-color: #1e3a8a; color: white; padding: 6px 16px; border-radius: 20px; font-size: 14px; font-weight: bold; }
    .cluster-header { background-color: #1e3a8a; color: white; padding: 8px; border-radius: 6px; margin-top: 15px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="names-frame">
        <div class="project-title">🔍 المنظومة المتكاملة لاسترجاع المعلومات والتجميع (2026)</div>
        <p style="text-align: center; color: #555; font-size: 16px; font-weight: bold;">تحت إشراف م. مروة الداية / م. سليمى</p>
        <div class="team-grid">
            <div class="member-badge">👩‍💻 مايا عرفات </div>
            <div class="member-badge">👩‍💻 لمى الحلقي</div>
            <div class="member-badge">👩‍💻 ماريا السقال </div>
            <div class="member-badge">👩‍💻 غزل عبد الواحد </div>
            <div class="member-badge">👩‍💻 سالي الاسعد </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# ==========================================================

from faiss_indexing_service import load_faiss_index, search_faiss_index

@st.cache_resource
def load_all_services_optimized():
    if os.path.exists("clustered_documents.pkl"):
        dataframe = pd.read_pickle("clustered_documents.pkl")
    elif os.path.exists("processed_documents.pkl"):
        dataframe = pd.read_pickle("processed_documents.pkl")
    else:
        st.error("❌ لم يتم العثور على ملفات البيانات المستخرجة.")
        st.stop()
        
    if "doc_id" not in dataframe.columns:
        dataframe["doc_id"] = dataframe.index

    bm25_file = "bm25_model.pkl"
    if os.path.exists(bm25_file):
        with open(bm25_file, "rb") as f:
            cached_bm25_model = pickle.load(f)
    else:
        st.error("❌ ملف أصول المكتبة 'bm25_model.pkl' غير موجود! يرجى تشغيل 'search_core_service.py' أولاً.")
        st.stop()

    with st.spinner("⏳ جاري تحميل الأوزان وفهرس المتجهات FAISS المطور..."):
        bert_model_instance = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        try:
            faiss_index = load_faiss_index("faiss_index.bin")
        except Exception as e:
            st.error(f"❌ لم يتم العثور على ملف الفهرس الشعاعي 'faiss_index.bin'. التفاصيل: {e}")
            st.stop()
        
    doc_ids_list = dataframe["doc_id"].tolist()
    text_lookup_dict = dict(zip(dataframe["doc_id"], dataframe["cleaned_text"]))
    docid_to_matrix_row = {doc_id: idx for idx, doc_id in enumerate(doc_ids_list)}

    cached_freq_counter = Counter()
    with st.spinner("📊 جاري إحصاء الكلمات وتجهيز الإحصائيات بأمان..."):
        for text in dataframe["cleaned_text"].dropna().astype(str):
            cached_freq_counter.update(text.lower().split())

    return (
        dataframe, cached_bm25_model, bert_model_instance, faiss_index,
        doc_ids_list, text_lookup_dict, docid_to_matrix_row, cached_freq_counter
    )

(df, bm25_model, bert_model, faiss_index, 
 doc_ids, fast_text_lookup, docid_to_row, cached_freq) = load_all_services_optimized()

if 'word_freq' not in st.session_state:
    st.session_state.word_freq = cached_freq

# ==========================================================

with st.sidebar:
    st.markdown('<div class="sidebar-header-custom">⚙️ لوحة التحكم</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section-card"><p class="section-label">🧠 اختيار محرك الاسترجاع:</p>', unsafe_allow_html=True)
    search_type = st.radio(
        "المحركات المتاحة:",
        [
            "محرك BM25 التقليدي", 
            "محرك BERT الدلالي", 
            "نظام الترتيب المتسلسل Cascade (الهجين)",
            "نظام صهر الرتب الهجين (RRF)",
            "نموذج إعادة الترتيب الذكي Learning-to-Rank (LTR)"
        ],
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section-card"><p class="section-label">🎛️ معاملات تحسين BM25 الحية:</p>', unsafe_allow_html=True)
    k1 = st.slider("معامل تكرار اللفظ (k1):", min_value=1.2, max_value=2.0, value=1.5, step=0.1)
    b = st.slider("معامل طول الوثيقة (b):", min_value=0.5, max_value=0.9, value=0.75, step=0.05)
    st.markdown('</div>', unsafe_allow_html=True)

    if "Cascade" in search_type:
        st.markdown('<div class="sidebar-section-card"><p class="section-label">⛓️ إعدادات التصفية المتسلسلة:</p>', unsafe_allow_html=True)
        cascade_top_bm25 = st.slider("عدد مرشحي المرحلة الأولى (BM25):", min_value=50, max_value=1000, value=500, step=50)
        st.markdown('</div>', unsafe_allow_html=True)

    if "RRF" in search_type:
        st.markdown('<div class="sidebar-section-card"><p class="section-label">⚙️ معامل تلطيف RRF (k):</p>', unsafe_allow_html=True)
        rrf_k = st.slider("قيمة الثابت k لـ RRF:", min_value=10, max_value=100, value=60, step=5)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section-card"><p class="section-label">📝 معالجة الاستعلام (Query):</p>', unsafe_allow_html=True)
    enable_expansion = st.checkbox("تفعيل توسيع الاستعلام بـ WordNet", value=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================================

def api_gateway_complete(query, model_name, bm25_k1, bm25_b, expand_flag, rrf_k_val=60, cascade_top=500):
    with st.spinner("📝 خطوة 1: جاري تشغيل المعالجة وتوسيع الاستعلام بالـ NLP..."):
        processed_query = process_query_service(query, expand=expand_flag)
    
    formatted_results = []
    tokenized_query = processed_query.split()
    
    bm25_model.k1 = bm25_k1
    bm25_model.b = bm25_b

    if not tokenized_query:
        return [], processed_query

    if "BM25" in model_name:
        with st.spinner("🔢 خطوة 2: جاري احتساب أوزان وتكرارات الفهرس المعكوس لـ BM25..."):
            doc_scores = bm25_model.get_scores(tokenized_query)
            raw_zipped = list(zip(doc_ids, doc_scores))
            sorted_bm25 = sorted(raw_zipped, key=lambda x: x[1], reverse=True)[:10]
            
        for rank, (doc_id, score) in enumerate(sorted_bm25, 1):
            actual_text = fast_text_lookup.get(doc_id, "No text available.")
            formatted_results.append({"doc_id": doc_id, "score": round(score, 4), "text": str(actual_text)})
            
    elif "BERT" in model_name:
        with st.spinner("🧠 خطوة 2: جاري البحث الدلالي الفائق باستخدام FAISS..."):
            query_vec = bert_model.encode([processed_query])
            distances, top_indices = search_faiss_index(query_vec, faiss_index, top_k=10)
            
            distances = distances.flatten()
            top_indices = top_indices.flatten()
            
        for rank, idx in enumerate(top_indices):
            doc_id = doc_ids[idx]
            actual_text = fast_text_lookup.get(doc_id, "No text available.")
            formatted_results.append({
                "doc_id": doc_id, 
                "score": round(float(distances[rank]), 4), 
                "text": str(actual_text)
            })
            
    elif "Cascade" in model_name or "المتسلسل" in model_name:
        with st.spinner(f"⛓️ جاري استدعاء المرشحين وحساب المتجهات حياً لـ {cascade_top} مستند بأمان..."):
            doc_scores = bm25_model.get_scores(tokenized_query)
            raw_zipped = list(zip(doc_ids, doc_scores))
            candidate_docs = sorted(raw_zipped, key=lambda x: x[1], reverse=True)[:cascade_top]
            candidate_docs = [doc for doc in candidate_docs if doc[1] > 0]
            
            if candidate_docs:
                query_embedding = bert_model.encode([processed_query]).reshape(1, -1)
                candidate_texts = [fast_text_lookup.get(d_id, "") for d_id, _ in candidate_docs]
                candidate_embeddings = bert_model.encode(candidate_texts)
                
                similarity_vector = cosine_similarity(query_embedding, candidate_embeddings)[0]
                
                reranked_cascade = []
                for idx, (doc_id, _) in enumerate(candidate_docs):
                    reranked_cascade.append((doc_id, similarity_vector[idx]))
                
                reranked_cascade.sort(key=lambda x: x[1], reverse=True)
                
                for doc_id, score in reranked_cascade[:10]:
                    actual_text = fast_text_lookup.get(doc_id, "No text available.")
                    formatted_results.append({"doc_id": doc_id, "score": round(float(score), 4), "text": str(actual_text)})
        
    elif "صهر الرتب" in model_name or "RRF" in model_name:
        with st.spinner("🔗 نظام هجين RRF: جاري دمج مخرجات BM25 و كاش FAISS السريع..."):
            doc_scores = bm25_model.get_scores(tokenized_query)
            bm25_res_all = sorted(list(zip(doc_ids, doc_scores)), key=lambda x: x[1], reverse=True)[:100]
            
            query_vec = bert_model.encode([processed_query])
            distances, top_indices = search_faiss_index(query_vec, faiss_index, top_k=100)
            
            top_indices = top_indices.flatten()
            
            fused_scores = defaultdict(float)
            for rank, (doc_id, _) in enumerate(bm25_res_all, start=1):
                fused_scores[doc_id] += 1.0 / (rrf_k_val + rank)
                
            for rank, idx in enumerate(top_indices, start=1):
                doc_id = doc_ids[idx]
                fused_scores[doc_id] += 1.0 / (rrf_k_val + rank)
                
            fused_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:10]
            
        for rank, (doc_id, score) in enumerate(fused_results, 1):
            actual_text = fast_text_lookup.get(doc_id, "No text available.")
            formatted_results.append({"doc_id": doc_id, "score": round(score, 6), "text": str(actual_text)})

    elif "LTR" in model_name or "Learning-to-Rank" in model_name:
        with st.spinner("⚖️ جاري تشغيل نموذج الترتيب الذكي LTR واستخراج الميزات الناتجة حياً..."):
            try:
                from service_addLTR import ltr_ranked_search
                ltr_res = ltr_ranked_search(query, top_k=10)
            except Exception as e:
                st.error(f"❌ فشل استدعاء خدمة LTR المضافة. تأكد من وجود الملفات. التفاصيل: {e}")
                return [], processed_query
                
        for rank, (doc_id, score) in enumerate(ltr_res, 1):
            actual_text = fast_text_lookup.get(str(doc_id), "No text available.")
            formatted_results.append({
                "doc_id": doc_id, 
                "score": round(float(score), 4), 
                "text": str(actual_text)
            })

    return formatted_results, processed_query


# =====================================================================

st.write("---")
tab1, tab2 = st.tabs(["🔍 البحث والنتائج", "📁 المجموعات الصنفية (Clustering)"])

# ---------------------------------------------------------------------

with tab1:
    with st.form("search_form", clear_on_submit=False):
        user_query = st.text_input(
            "✍️ أدخل استعلامك باللغة الطبيعية للبحث:", 
            placeholder="مثال: diabetes treatment or high blood pressure", 
            key="search_input"
        )
        submit_button = st.form_submit_button("🚀 إطلاق المحرك المتكامل")

    current_rrf_k = rrf_k if "RRF" in search_type else 60
    current_cascade_top = cascade_top_bm25 if "Cascade" in search_type else 500

    if submit_button and user_query.strip() != "":
        with st.spinner("🔍 جاري تشغيل خطوط معالجة الـ API Gateway وسحب الفهارس..."):
            results, final_query = api_gateway_complete(
                user_query, search_type, k1, b, enable_expansion, 
                rrf_k_val=current_rrf_k, cascade_top=current_cascade_top
            )
            st.session_state.search_results = (results, final_query)

    # عرض النتائج في حال وجودها
    if "search_results" in st.session_state:
        results, final_query = st.session_state.search_results
        
        st.success("🎯 تمت عملية الاسترجاع بنجاح!")
        st.info(f"📝 النص بعد المعالجة والتوسيع: **{final_query}**")
        
        st.write("---")
        st.write(f"### 🗂️ قائمة الوثائق الـ 10 المسترجعة حالياً:")
        for i, res in enumerate(results, 1):
            if "RRF" in search_type:
                score_label = "RRF Score"
            elif "LTR" in search_type:
                score_label = "LTR Probability Score"
            elif "Cascade" in search_type or "BERT" in search_type:
                score_label = "Refined Semantic Score"
            else:
                score_label = "Cached BM25 Score"
                
            st.write(f"**الرتبة {i:02d}** | معرف الوثيقة: `{res['doc_id']}` | {score_label}: `{res['score']}`")
            st.code(res['text'], language="text")
    else:
        st.info("💡 اكتب أي استعلام في الأعلى واضغط على الإطلاق لتظهر لك الوثائق هنا.")

# ---------------------------------------------------------------------

with tab2:
    st.write("### 📁 تجميع وتصنيف الوثائق دلالياً (Document Clustering)")
    st.write("فرز قاعدة البيانات دلالياً باستخدام خوارزمية **KMeans بـ 10 مجموعات صنفية** لفحص الأسئلة المتشابهة في مواضيع موحدة يدوياً وتلقائياً.")
    
    st.markdown("#### 🧠 1️⃣ اختبار التنبؤ المباشر للمجموعة (Live Cluster Prediction):")
    cluster_input = st.text_input(
        "اكتب أي سؤال جديد هنا بلغة طبيعية ليتنبأ الموديل بمجموعته فوراً:", 
        placeholder="مثال: how to treat diabetes or high blood pressure...",
        key="cluster_prediction_field"
    )

    if cluster_input.strip() != "":
        query_embedding = bert_model.encode([cluster_input])
        predicted_id = None

        if os.path.exists("kmeans_model.pkl"):
            with open("kmeans_model.pkl", "rb") as f:
                kmeans_model = pickle.load(f)
            predicted_id = int(kmeans_model.predict(query_embedding)[0])
        
        elif os.path.exists("medical_embeddings.npy") and "cluster" in df.columns:
            embeddings_matrix = np.nan_to_num(np.load("medical_embeddings.npy"))
            centroids = []
            for c_id in range(10):
                cluster_vectors = embeddings_matrix[df["cluster"] == c_id]
                if len(cluster_vectors) > 0:
                    centroids.append(cluster_vectors.mean(axis=0))
                else:
                    centroids.append(np.zeros(embeddings_matrix.shape[1]))

            safe_centroids = np.nan_to_num(np.array(centroids))
            similarities = cosine_similarity(query_embedding, safe_centroids)[0]
            predicted_id = int(np.argmax(similarities))
        
        if predicted_id is not None:
            st.success(f"🎯 المجموعة المتوقعة لهذا النص هي: **Cluster {predicted_id}**")
        else:
            st.warning("⚠️ لا يمكن التنبؤ حالياً. يرجى التأكد من تشغيل سكربت التجميع وتوليد الملفات أولاً.")
                
    st.write("---")
    st.write("#### 📊 2️⃣ توزيع وحجم الوثائق لكل مجموعة (Cluster Sizes):")
    
    if "cluster" in df.columns:
        cluster_counts = df["cluster"].value_counts().sort_index()
        counts_df = pd.DataFrame({
            "رقم المجموعة (Cluster ID)": [f"Cluster {i}" for i in range(10)],
            "عدد الأسئلة داخلها (Document Count)": [cluster_counts.get(i, 0) for i in range(10)]
        })
        st.dataframe(counts_df, use_container_width=True)
        st.bar_chart(counts_df.set_index("رقم المجموعة (Cluster ID)"))
    else:
        st.info("💡 لم يتم العثور على عمود المجموعات (cluster) في البيانات. يرجى تشغيل سكربت التجميع أولاً.")
    
    st.write("---")
    st.write("#### 🔍 3️⃣ تصفح وفحص محتويات المجموعات الصنفية يدوياً:")
    selected_cluster = st.selectbox("اختر رقم المجموعة المراد معاينة نصوصها الحقيقية:", list(range(10)))
    filter_keyword = st.text_input("🔍 تصفية الأسئلة داخل هذه المجموعة بكلمة مفتاحية (اختياري):", placeholder="اكتب كلمة للبحث داخل الـ Cluster...")
    
    if selected_cluster is not None and "cluster" in df.columns:
        st.markdown(f'<div class="cluster-header">📄 نصوص المستندات في المجموعة الصنفية رقم {selected_cluster}:</div>', unsafe_allow_html=True)
        cluster_docs = df[df["cluster"] == selected_cluster]["cleaned_text"].astype(str)
        
        if filter_keyword.strip() != "":
            cluster_docs = cluster_docs[cluster_docs.str.contains(filter_keyword.lower(), case=False, na=False)]
            
        sample_docs = cluster_docs.head(10).tolist()
        if sample_docs:
            for idx, doc in enumerate(sample_docs, 1):
                st.markdown(f"**{idx}.** {doc}")
        else:
            st.info("⚠️ لا توجد نتائج مطابقة للكلمة المفتاحية داخل هذه المجموعة.")
    elif selected_cluster is not None:
        st.info("💡 لا يمكن عرض النصوص لعدم توليد عمود المجموعات في البيانات بعد.")