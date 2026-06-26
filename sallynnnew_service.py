import os
import re
import shutil
import nltk
import pandas as pd
from textblob import TextBlob
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# ==========================================
# 1. إدارة ملفات البيانات (Pickle Handling)
# ==========================================

file_path = r"C:\Users\ev\Documents\ir_project-main\processed_documents.pkl"
target_dir = os.path.dirname(file_path)

# التأكد من وجود المجلد الخاص بالملف
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# التحقق من مكان عمل Jupyter الحالي
print("المجلد الذي يعمل فيه البرنامج الآن هو:", os.getcwd())

# تحميل الملف أو إنشاء DataFrame تجريبي وحفظه
if os.path.exists(file_path):
    print("📁 جاري تحميل الملف من المسار المحدد...")
    df = pd.read_pickle(file_path)
    print("✅ تم تحميل البيانات بنجاح!")
else:
    print("⚠️ الملف غير موجود. يتم الآن إنشاء DataFrame تجريبي وحفظه...")
    data = {
        "cleaned_text": [
            "titl congenit adren hyperplasia calcium channe",
            "titl lead burden alter neuropsycholog develop ",
            "titl vaccin tetanu klh assess immun respons co",
            "titl degre centigrad whole bodi hyperthermia t",
            "titl bodi water content cyanot congenit heart "
        ]
    }
    df = pd.DataFrame(data, columns=["cleaned_text"])
    df.to_pickle(file_path)
    print(f"✅ تم حفظ ملف الـ Pickle الجديد في: {file_path}")

print("\n--- عينة من البيانات ---")
print(df.head())
print("-----------------------\n")


# ==========================================
# 2. أدوات المعالجة المسبقة (Preprocessing)
# ==========================================

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def preprocess_query(query):
    query = query.lower()
    query = re.sub(r'[^a-zA-Z\s]', '', query)
    tokens = word_tokenize(query)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)

def process_query_service(query, expand=True):
    query_clean = re.sub(r'[^a-z\s]', '', query.lower())
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


# ==========================================
# 3. توسيع الاستعلام وتصحيح الأخطاء الإملائية
# ==========================================

def get_synonyms(word):
    synonyms = set()
    synsets = wordnet.synsets(word)[:2]

    for syn in synsets:
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ").lower()
            if synonym == word.lower():
                continue
            if not synonym.isalpha():
                continue
            if len(synonym) < 3:
                continue
            synonyms.add(synonym)

    return list(synonyms)

def expand_query(query):
    words = query.lower().split()
    expanded = words.copy()
    for word in words:
        expanded.extend(get_synonyms(word)[:2])
    return " ".join(set(expanded))

def suggest_query(query):
    return str(TextBlob(query).correct())

def query_service_v3(query):
    corrected_query = str(TextBlob(query).correct())
    processed_query = process_query_service(corrected_query)
    expanded_query = expand_query(corrected_query)

    return {
        "original_query": query,
        "corrected_query": corrected_query,
        "processed_query": processed_query,
        "expanded_query": expanded_query
    }


# ==========================================
# 4. التنفيذ والتشغيل التجريبي
# ==========================================

print("جاري معالجة استعلام تجريبي ('hart diseas treatment')...")
result = query_service_v3("hart diseas treatment")
print(result)
print("\n" + "="*50 + "\n")

queries = [
    "Heart disease treatment",
    "Breast cancer therapy",
    "Diabetes clinical trial",
    "Congenital heart disease",
    "Calcium channel blockers",
    "Body water content",
    "Neuropsychological development",
    "Immune response vaccination",
    "Whole body hyperthermia",
    "Lead burden effects"
]

print("تشغيل الخدمة على قائمة الاستعلامات:")
for q in queries:
    print(query_service_v3(q))
    print("-" * 50)