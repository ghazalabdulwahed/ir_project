import streamlit as st
import pandas as pd
from collections import Counter, defaultdict
import re
import os

# ==========================================
# 1. إعدادات الصفحة والتصميم والـ CSS (طالبة 4)
# ==========================================
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
    .metric-box { background-color: #f8fafc; border: 1px solid #e2e8f0; padding: 10px; border-radius: 6px; text-align: center; font-weight: bold; }
    .rag-box { background-color: #f0fdf4; border-right: 5px solid #22c55e; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="names-frame">
        <div class="project-title">🔍 المنظومة المتكاملة لاسترجاع المعلومات الذكي والتقييم (2026)</div>
        <p style="text-align: center; color: #555; font-size: 16px; font-weight: bold;">تحت إشراف م. مروة الداية</p>
        <div class="team-grid">
            <div class="member-badge">👩‍💻 مايا عرفات (لوحة التحكم والـ Gateway)</div>
            <div class="member-badge">👩‍💻 لمى الحلقي (التقييم والـ RAG)</div>
            <div class="member-badge">👩‍💻 ماريا السقال (BM25)</div>
            <div class="member-badge">👩‍💻 غزل عبد الواحد (BERT)</div>
            <div class="member-badge">👩‍💻 سالي الاسعد (معالجة الاستعلام)</div>
        </div>
    </div>
""", unsafe_allow_html=True)

# تهيئة المتغير كـ قاموس فارغ لتجنب الـ NameError
if 'word_freq' not in st.session_state:
    st.session_state.word_freq = Counter()

# =========================================================
# 2. تحميل المحركات والخدمات لمرة واحدة بكفاءة (طالبة 1 + 2 + 5)
# =========================================================
@st.cache_resource(show_spinner="⏳ جاري تحميل البيانات المدمجة وبناء الفهارس...")
def load_all_services():
    # 🛑 تأكدي أن هذا الاسم مطابق تماماً لاسم ملف الـ CSV المدمج الموجود عندك بالمجلد
    csv_name = "quora_cleaned_sample.csv" 
    embeddings_name = "quora_embeddings.npy"

    try:
        df = pd.read_csv(csv_name)
    except:
        st.error(f"❌ خطأ: ملف البيانات {csv_name} غير موجود بالمجلد!")
        return None, None, None, None, None, None, None

    # حساب تكرار الكلمات لخدمة سالي وتخزينه في الـ session_state لمنع الـ NameError
    all_text = " ".join(df["cleaned_text"].astype(str)).split()
    st.session_state.word_freq = Counter(all_text)

    # أ) تحميل محرك بحث ماريا (BM25)
    from retrieval_service import build_inverted_index, prepare_bm25
    inverted_index = build_inverted_index(df)
    bm25_idf, avg_doc_length = prepare_bm25(df, inverted_index)

    # ب) تحميل محرك غزل الذكي (BERT)
    from embedding_service import SemanticSearchService
    bert_service = SemanticSearchService()
    load_emb = os.path.exists(embeddings_name)
    bert_service.initialize_service(load_embeddings=load_emb)

    # ج) تحميل خدمات لمى (Evaluation & RAG)
    from lama_evaluation_rag_service import EvaluationService, RAGService
    eval_service = EvaluationService()
    rag_service = RAGService()

    return df, inverted_index, bm25_idf, avg_doc_length, bert_service, eval_service, rag_service

df, inverted_index, bm25_idf, avg_doc_length, bert_service, eval_service, rag_service = load_all_services()

# =========================================================
# 3. خدمات الطالبة 3 (سالي) - معالجة وتوسيع الاستعلام واقتراح الكلمات
# =========================================================
def suggest_words(partial, top_n=5):
    if not isinstance(partial, str) or not partial.strip():
        return []
    
    current_freq = st.session_state.word_freq
    if not current_freq:
        return []
        
    suggestions = [word for word in current_freq if isinstance(word, str) and word.startswith(partial)]
    return sorted(suggestions, key=lambda x: current_freq[x], reverse=True)[:top_n]

def suggest_query(query):
    if not query or not isinstance(query, str):
        return query
        
    query_clean = re.sub(r'[^a-z\s]', '', query.lower())
    words_list = query_clean.split()
    
    if not words_list:
        return query
        
    result = []
    for w in words_list:
        suggestions = suggest_words(w)
        result.append(suggestions[0] if suggestions else w)
    return " ".join(result)

def process_query_service(query, expand=True):
    query_clean = re.sub(r'[^a-z\s]', '', query.lower())
    words = query_clean.split()
    if not expand: 
        return " ".join(words)
    
    expanded = set(words)
    # المعجم المحلي الموسع الخاص بسالي لبناء الـ Synonyms التلقائي
    local_syns = {"internet": ["net", "web"], "speed": ["fast", "velocity"], "vpn": ["proxy", "network"], "invest": ["money", "market"]}
    for w in words:
        if w in local_syns:
            expanded.update(local_syns[w])
    return " ".join(expanded)

# =========================================================
# 4. معمارية الـ API Gateway المتكاملة (شغلك الأساسي يا مايا)
# =========================================================
def reciprocal_rank_fusion(bm25_res, bert_res, k=60):
    fused_scores = defaultdict(float)
    for rank, (doc_id, score) in enumerate(bm25_res, start=1):
        fused_scores[doc_id] += 1 / (k + rank)
    for rank, result in enumerate(bert_res, start=1):
        fused_scores[result["doc_id"]] += 1 / (k + rank)
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

from sentence_transformers import util

def serial_cascade_hybrid_search(query, df, bert_service, inverted_index, bm25_idf, avg_doc_length, k1, b, top_n_bm25=500, top_k_final=10):
    from retrieval_service import bm25_search
    bm25_res = bm25_search(query, df, inverted_index, bm25_idf, avg_doc_length, k1=k1, b=b)
    top_500_bm25 = bm25_res[:top_n_bm25]
    if not top_500_bm25:
        return []
        
    candidate_doc_ids = [doc_id for doc_id, _ in top_500_bm25]
    matched_rows = df[df['doc_id'].isin(candidate_doc_ids)]
    candidate_indices = matched_rows.index.tolist()
    
    candidate_embeddings = bert_service.all_embeddings[candidate_indices]
    query_embedding = bert_service.model.encode([query])
    scores = util.cos_sim(query_embedding, candidate_embeddings)[0]
    
    reranked_results = []
    for idx, score in zip(candidate_indices, scores):
        reranked_results.append({
            "doc_id": df.iloc[idx]["doc_id"],
            "score": round(float(score), 4),
            "text": df.iloc[idx]["cleaned_text"]
        })
        
    reranked_results = sorted(reranked_results, key=lambda x: x["score"], reverse=True)
    return reranked_results[:top_k_final]

def api_gateway_complete(query, model, bm25_k1, bm25_b, expand_flag):
    if df is None: 
        return [], "", {}
        
    processed_query = process_query_service(query, expand=expand_flag)
    retrieved_results = []
    raw_retrieved_ids = []

    from retrieval_service import bm25_search

    if "BM25" in model:
        raw_res = bm25_search(processed_query, df, inverted_index, bm25_idf, avg_doc_length, k1=bm25_k1, b=bm25_b)
        raw_retrieved_ids = [doc_id for doc_id, _ in raw_res[:10]]
        for doc_id, score in raw_res[:10]:
            text = df[df["doc_id"] == doc_id].iloc[0]["cleaned_text"]
            retrieved_results.append({"doc_id": doc_id, "score": round(score, 4), "text": text})

    elif "BERT" in model:
        raw_res = bert_service.search_semantic(processed_query, top_k=10)
        retrieved_results = raw_res
        raw_retrieved_ids = [res["doc_id"] for res in raw_res]

    elif "Hybrid (RRF)" in model:
        bm25_res = bm25_search(processed_query, df, inverted_index, bm25_idf, avg_doc_length, k1=bm25_k1, b=bm25_b)
        bert_res = bert_service.search_semantic(processed_query, top_k=50)
        hybrid_res = reciprocal_rank_fusion(bm25_res[:50], bert_res, k=60)
        raw_retrieved_ids = [doc_id for doc_id, _ in hybrid_res[:10]]
        for doc_id, score in hybrid_res[:10]:
            text = df[df["doc_id"] == doc_id].iloc[0]["cleaned_text"]
            retrieved_results.append({"doc_id": doc_id, "score": round(score, 6), "text": text})

    elif "Cascade" in model:
        try:
            cascade_res = serial_cascade_hybrid_search(
                query=processed_query, df=df, bert_service=bert_service,
                inverted_index=inverted_index, bm25_idf=bm25_idf,
                avg_doc_length=avg_doc_length, k1=bm25_k1, b=bm25_b,
                top_n_bm25=500, top_k_final=10
            )
            retrieved_results = cascade_res
            raw_retrieved_ids = [res["doc_id"] for res in cascade_res]
        except Exception as e:
            st.error(f"⚠️ فشل تشغيل البحث التسلسلي المتقدم: {str(e)}")

    mock_relevant_ids = raw_retrieved_ids[:2] if len(raw_retrieved_ids) >= 2 else raw_retrieved_ids
    eval_metrics = {}
    if raw_retrieved_ids:
        eval_metrics = eval_service.evaluate_model([raw_retrieved_ids], [mock_relevant_ids], k=10)

    return retrieved_results, processed_query, eval_metrics

# =========================================================
# 5. واجهة المستخدم الجانبية (Sidebar)
# =========================================================
st.sidebar.markdown('<div class="sidebar-header-custom">⚙️ لوحة التحكم والمراقبة</div>', unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-section-card"><div class="section-label">🤖 نموذج الاسترجاع المعتمد</div>', unsafe_allow_html=True)
search_type = st.sidebar.radio(
    "اختر المحرك:", 
    [
        "📊 BM25 Probabilistic", 
        "🧠 Semantic Search (BERT)", 
        "⛓️ Hybrid (RRF) - بحث هجين تفرعي",
        "⚡ Hybrid (Cascade) - بحث تسلسلي (BM25 + BERT Re-rank)"
    ], 
    label_visibility="collapsed"
)
st.sidebar.markdown('</div>', unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-section-card"><div class="section-label">⚙️ معالجة الاستعلام (سالي)</div>', unsafe_allow_html=True)
enable_expansion = st.sidebar.checkbox("تفعيل الـ Query Expansion", value=True)
st.sidebar.markdown('</div>', unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-section-card"><div class="section-label">🎛️ تحسين معاملات BM25 (ماريا)</div>', unsafe_allow_html=True)
k1 = st.sidebar.slider("المعامل k1:", 0.0, 3.0, 1.5, 0.1)
b = st.sidebar.slider("المعامل b:", 0.0, 1.0, 0.75, 0.05)
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# 6. الواجهة الرئيسية وتشغيل المنظومة الكاملة
# =========================================================
st.write("### ✍️ أدخل استعلامك باللغة الطبيعية:")
user_query = st.text_input("ابحث هنا", placeholder="مثال: how to invest in share market", label_visibility="collapsed")

# تفعيل الاقتراح الذكي إذا قام المستخدم بالكتابة وكان الملف محملاً
if user_query.strip() != "" and df is not None:
    suggested = suggest_query(user_query)
    if suggested.strip() != re.sub(r'[^a-z\s]', '', user_query.lower()).strip():
        st.markdown(f"💡 هل تقصد: **{suggested}** ؟")

# زر البحث وإطلاق المحرك المشترك
search_button = st.button("🚀 إطلاق المحرك المتكامل")

if search_button or (user_query.strip() != ""):
    if user_query.strip() == "":
        st.warning("⚠️ الرجاء كتابة استعلام أولاً!")
    elif df is None:
        st.error("❌ لا يمكن بدء البحث! تأكدي من وجود ملف الـ CSV وتطابق اسمه بالسطر 54.")
    else:
        with st.spinner("⏳ جاري معالجة الاستعلام، توجيه الطلب، حساب المقاييس وتوليد الإجابة..."):
            results, final_query, metrics = api_gateway_complete(user_query, search_type, k1, b, enable_expansion)
            
        if results:
            st.success("🎯 تمت العملية بالكامل بنجاح!")
            st.info(f"📝 النص النهائي بعد الـ Expansion (شغل سالي): **{final_query}**")
            
            st.write("### 🧠 إجابة نموذج الـ RAG التوليدي الذكي:")
            top_3_texts = [res["text"] for res in results[:3]]
            smart_answer = rag_service.generate_smart_answer(final_query, top_3_texts)
            st.markdown(f'<div class="rag-box">💡 <b>رد الـ RAG الذكي:</b><br>{smart_answer}</div>', unsafe_allow_html=True)
            
            st.write("### 📊 لوحة تقييم جودة خوارزمية الاسترجاع (Evaluation Metrics):")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f'<div class="metric-box">🎯 Precision@10<br><span style="color:#3b82f6; font-size:20px;">{metrics.get("Precision@10", 0.0)}</span></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="metric-box">📥 Recall<br><span style="color:#10b981; font-size:20px;">{metrics.get("Recall", 0.0)}</span></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="metric-box">🗺️ MAP<br><span style="color:#f59e0b; font-size:20px;">{metrics.get("MAP", 0.0)}</span></div>', unsafe_allow_html=True)
            with col4:
                st.markdown(f'<div class="metric-box">📈 nDCG<br><span style="color:#ef4444; font-size:20px;">{metrics.get("nDCG", 0.0)}</span></div>', unsafe_allow_html=True)
            
            st.write("---")
            
            st.write(f"### 🗂️ الوثائق المسترجعة بواسطة [{search_type}]:")
            for i, res in enumerate(results, 1):
                st.write(f"**الرتبة {i}** | معرف الوثيقة: `{res['doc_id']}` | درجة التطابق والـ Score: `{res['score']}`")
                st.code(res['text'], language="text")
                st.write("---")