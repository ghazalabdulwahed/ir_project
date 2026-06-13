import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_query(text):
    text = text.lower()  # تصغير الحروف
    text = re.sub(r'[^a-z\s]', '', text)  # حذف الرموز والأرقام
    words = word_tokenize(text)  # تقسيم الكلمات
    words = [w for w in words if w not in stop_words]  # حذف stopwords
    return " ".join(words)
query = "What is the step by step guide to invest in share market?"
cleaned_query = clean_query(query)

print(cleaned_query)
import nltk
from nltk.corpus import wordnet
nltk.download('wordnet')
def refine_query(query):
    cleaned = clean_query(query)
    words = cleaned.split()
    
    expanded_query = set(words)
    
    for w in words:
        expanded_query.update(get_synonyms(w))
    
    return " ".join(expanded_query)
query = "internet speed vpn"
print(refine_query(query))

import nltk
from nltk.corpus import wordnet

nltk.download('wordnet')

query = "internet speed vpn"
print(refine_query(query))

from nltk.corpus import wordnet

def get_synonyms(word):
    synonyms = set()
    
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ")
            synonyms.add(synonym.lower())
    
    return list(synonyms)
print(get_synonyms("internet"))

import pandas as pd
from collections import Counter

df = pd.read_csv("quora_cleaned_sample.csv")

all_text = " ".join(df["cleaned_text"].astype(str))
vocab = all_text.split()

word_freq = Counter(vocab)

import pandas as pd

df = pd.read_csv(r"C:\quora_cleaned_sample.csv")

import pandas as pd
from collections import Counter
import re

df = pd.read_csv(r"C:\quora_cleaned_sample.csv")

all_text = " ".join(df["cleaned_text"].astype(str))
words = all_text.split()

word_freq = Counter(words)
def suggest_words(partial, top_n=5):
    suggestions = []
    
    for word in word_freq:
        if word.startswith(partial):
            suggestions.append(word)
    
    suggestions = sorted(suggestions, key=lambda x: word_freq[x], reverse=True)
    
    return suggestions[:top_n]

def suggest_query(query):
    query = query.lower()
    query = re.sub(r'[^a-z\s]', '', query)
    
    words = query.split()
    
    result = []
    
    for w in words:
        suggestions = suggest_words(w)
        
        if suggestions:
            result.append(suggestions[0])
        else:
            result.append(w)
    
    return " ".join(result)
query = "intenet spedd vpn"

print("Original:", query)
print("Suggested:", suggest_query(query))

import pandas as pd

df = pd.read_csv(r"C:\quora_cleaned_sample.csv")  # عدلي المسار إذا مختلف

print(df.shape)
df.head()

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words("english"))

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace("_", " "))
    return list(synonyms)

def clean_query(query):
    tokens = word_tokenize(query.lower())
    return [w for w in tokens if w.isalnum() and w not in stop_words]

def process_query(query, expand=True):
    words = clean_query(query)

    if not expand:
        return " ".join(words)

    expanded = set(words)

    for w in words:
        expanded.update(get_synonyms(w))

    return " ".join(expanded)

# TEST
query = "internet speed vpn"

print("Original:", query)
print("Processed:", process_query(query))

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words("english"))

# -------------------------
# Clean Query
# -------------------------
def clean_query(query):
    tokens = word_tokenize(query.lower())
    return [w for w in tokens if w.isalnum() and w not in stop_words]

# -------------------------
# Synonyms (FILTERED)
# -------------------------
def get_synonyms(word, limit=3):
    synonyms = set()

    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            w = lemma.name().replace("_", " ")

            # فلترة: نأخذ كلمات قصيرة فقط ومفيدة
            if w.isalpha() and len(w) < 15:
                synonyms.add(w)

            if len(synonyms) >= limit:
                break

        if len(synonyms) >= limit:
            break

    return list(synonyms)

# -------------------------
# Main Query Processing
# -------------------------
def process_query(query, expand=True):
    words = clean_query(query)

    if not expand:
        return " ".join(words)

    expanded = set(words)

    for w in words:
        expanded.update(get_synonyms(w))

    return " ".join(expanded)


# -------------------------
# TEST
# -------------------------
query = "internet speed vpn"

print("Original:", query)
print("Processed:", process_query(query))