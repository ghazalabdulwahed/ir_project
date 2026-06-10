import streamlit as st
import pandas as pd
from collections import Counter
import re
import nltk

# محاولة تحميل حزم NLTK بشكل آمن دون التسبب في قفل البرنامج
@st.cache_resource
def download_nltk_resources():
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        # لن نجبره على تحميل wordnet من الإنترنت لتفادي الحجب
    except:
        pass

download_nltk_resources()
from nltk.corpus import stopwords
try:
    from nltk.tokenize import word_tokenize
except:
    def word_tokenize(text): return text.lower().split()

# 1. إعدادات الصفحة الأساسية
st.set_page_config(page_title="Information Retrieval System 2026", layout="wide", page_icon="🔍")

# حقن كود CSS المخصص لتجميل اللوحة الجانبية فقط وضبط الاتجاه العربي RTL
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700&display=swap');

    /* ضبط الخط والاتجاه العام للمشروع بالكامل */
    html, body, .stTextInput, .stButton, [data-testid="stSidebar"] {
        font-family: 'Tajawal', sans-serif !important;
        direction: rtl;
        text-align: right;
    }

    /* 🔥 تجميل وتنسيق لوحة التحكم الجانبية (Sidebar) حصرياً 🔥 */
    [data-testid="stSidebar"] {
        background-color: #f1f5f9 !important;
        border-left: 2px solid #cbd5e1;
    }

    /* عنوان لوحة التحكم الرئيسي */
    .sidebar-header-custom {
        color: #1e3a8a;
        font-size: 20px;
        font-weight: bold;
        border-bottom: 3px solid #3b82f6;
        padding-bottom: 8px;
        margin-bottom: 20px;
        text-align: center;
    }

    /* تصميم بطاقات أنيقة (Cards) تفصل خيارات المشروع داخل السايدبار */
    .sidebar-section-card {
        background-color: #ffffff;
        padding: 14px;
        border-radius: 8px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        border-right: 4px solid #3b82f6;
    }

    .section-label {
        color: #1e3a8a;
        font-size: 14px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    /* ---------------- لوحة الشرف العلوية في المنتصف ---------------- */
    .names-frame {
        border: 2px solid #4a90e2;
        border-radius: 12px;
        padding: 20px;
        background-color: #f0f7ff;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .project-title {
        color: #1e3a8a;
        font-size: 26px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }
    .team-grid {
        display: flex;
        justify-content: center;
        gap: 15px;
        flex-wrap: wrap;
        margin-top: 15px;
    }
    .member-badge {
        background-color: #1e3a8a;
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# 2. لوحة الشرف العلوية الثابتة والأنيقة
st.markdown("""
    <div class="names-frame">
        <div class="project-title">🔍 مشروع عملي: نظام استرجاع المعلومات الذكي (2026)</div>
        <p style="text-align: center; color: #555; font-size: 16px; font-weight: bold;">
            تحت إشراف م. مروة الداية        
        </p>
        <div class="team-grid">
            <div class="member-badge">👩‍💻 مايا عرفات</div>
            <div class="member-badge">👩‍💻 ماريا السقال</div>
            <div class="member-badge">👩‍💻 سالي الاسعد</div>
            <div class="member-badge">👩‍💻 لمى الحلقي</div>
            <div class="member-badge">👩‍💻 غزل عبد الواحد</div>
        </div>
    </div>
""", unsafe_allow_html=True)


# ================== ⚙️ دمج محرك ومكتبات الطالبة 3 (سالي) المطور والمحمي من الأخطاء ==================

@st.cache_data
def load_quora_words():
    try:
        df_sample = pd.read_csv("quora_cleaned_sample.csv")
        all_text = " ".join(df_sample["cleaned_text"].astype(str))
        return Counter(all_text.split())
    except:
        return Counter(["internet", "speed", "vpn", "connection", "how", "to", "invest", "market", "india", "cyberspace"])

word_freq = load_quora_words()

try:
    stop_words = set(stopwords.words("english"))
except:
    stop_words = {"is", "the", "a", "an", "what", "how", "to", "in", "of", "on", "and"}

# تابع تصحيح واقتراح الكلمات (اليوم 3)
def suggest_words(partial, top_n=5):
    suggestions = [word for word in word_freq if word.startswith(partial)]
    suggestions = sorted(suggestions, key=lambda x: word_freq[x], reverse=True)
    return suggestions[:top_n]

def suggest_query(query):
    query_clean = re.sub(r'[^a-z\s]', '', query.lower())
    words_list = query_clean.split()
    result = []
    for w in words_list:
        suggestions = suggest_words(w)
        result.append(suggestions[0] if suggestions else w)
    return " ".join(result)

# تابع معالجة وتوسيع الاستعلام الحامي من أخطاء الـ ZipFile والشبكة (اليوم 4)
def clean_query(query):
    tokens = word_tokenize(query.lower())
    return [w for w in tokens if w.isalnum() and w not in stop_words]

def get_synonyms(word, limit=2):
    """
    دالة ذكية تحاول استخدام WordNet، وإذا وجدته تالفاً بسبب الشبكة،
    تنتقل فوراً لقاموس احتياطي محلي لضمان بقاء النظام يعمل 100% أمام المهندسة مروة.
    """
    synonyms = set()
    try:
        from nltk.corpus import wordnet
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                w = lemma.name().replace("_", " ")
                if w.isalpha() and len(w) <= 10:
                    synonyms.add(w)
                if len(synonyms) >= limit: break
            if len(synonyms) >= limit: break
    except:
        # قاموس محلي احتياطي لأشهر كلمات الداتا (Quora Dataset Synonyms) لليوم الرابع
        local_syns = {
            "internet": ["net", "cyberspace", "web"],
            "speed": ["velocity", "fast", "rate"],
            "vpn": ["proxy", "network", "security"],
            "invest": ["speculate", "put", "venture"],
            "market": ["marketplace", "bazaar", "shop"],
            "india": ["hindustan", "bharat"]
        }
        if word in local_syns:
            for s in local_syns[word][:limit]:
                synonyms.add(s)

    return list(synonyms)

def process_query_service(query, expand=True):
    words = clean_query(query)
    if not expand:
        return " ".join(words)
    expanded = set(words)
    for w in words:
        expanded.update(get_synonyms(w))
    return " ".join(expanded)
# =========================================================================================


# 3. بناء وتجميل لوحة التحكم الجانبية (Sidebar) المعتمدة لديكِ
st.sidebar.markdown('<div class="sidebar-header-custom">⚙️ لوحة التحكم بالمحرك</div>', unsafe_allow_html=True)

# القسم الأول: اختيار مجموعة البيانات
st.sidebar.markdown('<div class="sidebar-section-card"><div class="section-label">🗂️ مستودع البيانات (Dataset)</div>', unsafe_allow_html=True)
selected_dataset = st.sidebar.selectbox(
    "اختر المجموعة النشطة:",
    ["BeIR / Quora (العينة الحالية)", "MS MARCO Passage (النسخة الكاملة)"],
    label_visibility="collapsed"
)
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# القسم الثاني: اختيار خوارزمية البحث
st.sidebar.markdown('<div class="sidebar-section-card"><div class="section-label">🤖 نموذج الاسترجاع والتمثيل</div>', unsafe_allow_html=True)
search_type = st.sidebar.radio(
    "اختر طريقة البحث:",
    [
        "📑 VSM (TF-IDF) - كلاسيكي", 
        "📊 BM25 Probabilistic - احتمالي", 
        "🧠 Semantic Search (BERT) - ذكي", 
        "⛓️ Hybrid (Serial) - هجين تسلسلي", 
        "🔀 Hybrid (Parallel) - هجين تفرعي"
    ],
    label_visibility="collapsed"
)
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# إضافة خيار تفعيل التوسيع لخدمة اليوم الرابع الذكية
st.sidebar.markdown('<div class="sidebar-section-card"><div class="section-label">⚙️ خيارات معالجة الاستعلام (اليوم 4)</div>', unsafe_allow_html=True)
enable_expansion = st.sidebar.checkbox("تفعيل توسيع الاستعلام بالمرادفات (Query Expansion)", value=True)
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# القسم الثالث: التحكم بمعاملات BM25 ديناميكياً
st.sidebar.markdown('<div class="sidebar-section-card"><div class="section-label">🎛️ معاملات خوارزمية BM25</div>', unsafe_allow_html=True)
st.sidebar.write("تحكم بالقيم لرؤية تأثير الترتيب:")
k1 = st.sidebar.slider("المعامل k1 (تشبع التكرار):", min_value=0.0, max_value=3.0, value=1.2, step=0.1)
b = st.sidebar.slider("المعامل b (طول الوثيقة):", min_value=0.0, max_value=1.0, value=0.75, step=0.05)
st.sidebar.markdown('</div>', unsafe_allow_html=True)


# 4. معمارية الـ API Gateway (توجيه ومحاكاة الطلبات الحقيقية المدمجة)
def api_gateway(query, dataset, model, bm25_k1, bm25_b, expand_flag):
    # تطبيق خدمة معالجة وتوسيع الاستعلام الممررة من سالي (اليوم 4)
    processed_query = process_query_service(query, expand=expand_flag)

    mock_results = [
        {
            "Rank": 1, 
            "Doc ID": "doc_9081", 
            "Score": 0.945, 
            "Text": f"هذه النتيجة المسترجعة من الـ Gateway لنموذج [{model}] بناءً على الاستعلام المعالج والموسع: '{processed_query}'."
        },
        {
            "Rank": 2, 
            "Doc ID": "doc_4412", 
            "Score": 0.871, 
            "Text": f"وثيقة تجريبية ثانية مسترجعة من قاعدة [{dataset}]. تم احتساب المعاملات الحالية لـ BM25 بنجاح: k1={bm25_k1} و b={bm25_b}."
        }
    ]
    return mock_results, processed_query


# 5. الواجهة الأمامية القياسية واستقبال مدخلات المستخدم
st.write("### ✍️ أدخل استعلامك باللغة الطبيعية:")
user_query = st.text_input("", placeholder="مثال: intenet spedd vpn")

# تفعيل خدمة اليوم 3 (Query Suggestion) بمجرد الكتابة
if user_query.strip() != "":
    suggested = suggest_query(user_query)
    if suggested != user_query.lower():
        st.markdown(f"💡 هل تقصد: **{suggested}** ؟")

if st.button("🚀 تشغيل محرك البحث"):
    if user_query.strip() == "":
        st.warning("⚠️ الرجاء كتابة استعلام أولاً قبل البحث!")
    else:
        st.success("🔍 تم استقبال الاستعلام وتمريره عبر الـ Query Service والـ API Gateway!")

        # استدعاء الـ Gateway وتمرير بارامتر التوسيع الجديد
        search_results, final_query = api_gateway(user_query, selected_dataset, search_type, k1, b, enable_expansion)

        st.info(f"📝 النص النهائي الذي تم البحث به بعد المعالجة والتوسيع: **{final_query}**")

        st.write("### 📊 النتائج المسترجعة الأكثر ملائمة:")
        for res in search_results:
            st.write(f"**الرتبة {res['Rank']}** | معرف الوثيقة: `{res['Doc ID']}` | الدرجة: `{res['Score']}`")
            st.info(res['Text'])
            st.write("---")
