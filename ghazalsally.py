#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# In[2]:


nltk.download('punkt')
nltk.download('stopwords')


# In[3]:


df = pd.read_csv("quora_cleaned_sample.csv")

df.head()


# In[4]:


from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

print(len(stop_words))


# In[5]:


from nltk.tokenize import word_tokenize
import string

def process_query(query):

    tokens = word_tokenize(query)

    filtered_tokens = [
        word.lower()
        for word in tokens
        if word.lower() not in stop_words
        and word not in string.punctuation
    ]

    return filtered_tokens


# In[6]:


query = "How can I learn machine learning quickly?"

print(process_query(query))


# In[ ]:





# In[7]:


def query_service(query):

    processed_query = process_query(query)

    return {
        "original_query": query,
        "processed_query": processed_query
    }


# In[8]:


result = query_service("How can I learn machine learning quickly?")

print(result)


# In[9]:


queries = [
    "What is Artificial Intelligence?",
    "How to learn Python fast?",
    "Best books for Data Science"
]

for q in queries:
    print(query_service(q))
    print("-" * 50)


# In[10]:


from nltk.corpus import wordnet

nltk.download('wordnet')


# In[11]:


def get_synonyms(word):

    synonyms = set()

    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ")

            if synonym.lower() != word.lower():
                synonyms.add(synonym.lower())

    return list(synonyms)


# In[12]:


print(get_synonyms("car")[:10])


# In[13]:


def expand_query(query):

    processed_query = process_query(query)

    expanded_terms = set(processed_query)

    for word in processed_query:

        synonyms = get_synonyms(word)

        expanded_terms.update(synonyms[:3])

    return list(expanded_terms)


# In[14]:


print(expand_query("car insurance"))


# In[15]:


get_ipython().system('pip install textblob')


# In[16]:


from textblob import TextBlob


# In[17]:


def correct_query(query):

    corrected_query = str(TextBlob(query).correct())

    return corrected_query


# In[18]:


print(correct_query("machne learnng"))


# In[19]:


def query_service_v3(query):

    # 1. Spell Correction
    corrected_query = str(TextBlob(query).correct())

    # 2. Processing (tokenization + stopwords)
    processed_query = process_query(corrected_query)

    # 3. Synonym Expansion
    expanded_query = expand_query(corrected_query)

    return {
        "original_query": query,
        "corrected_query": corrected_query,
        "processed_query": processed_query,
        "expanded_query": expanded_query
    }


# In[20]:


result = query_service_v3("machne learnng")

print(result)


# In[21]:


def get_synonyms(word):

    synonyms = set()

    for syn in wordnet.synsets(word):

        # خذ أول معنى فقط (يحسن الجودة)
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ")

            if synonym.lower() != word.lower():
                synonyms.add(synonym.lower())

        break  # 👈 هذا أهم تعديل

    return list(synonyms)


# In[22]:


print(get_synonyms("machine"))


# In[23]:


def get_synonyms(word):

    synonyms = set()

    synsets = wordnet.synsets(word)[:3]  # 👈 أهم تعديل

    for syn in synsets:
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ")

            if synonym.lower() != word.lower():
                synonyms.add(synonym.lower())

    return list(synonyms)


# In[24]:


print(get_synonyms("machine"))
print(get_synonyms("car"))
print(get_synonyms("learning"))


# In[25]:


def get_synonyms(word):

    synonyms = set()

    synsets = wordnet.synsets(word)[:5]

    for syn in synsets:
        for lemma in syn.lemmas():

            synonym = lemma.name().replace("_", " ").lower()

            # ❌ استبعاد نفس الكلمة
            if synonym == word.lower():
                continue

            # ❌ استبعاد الكلمات الطويلة الغريبة
            if len(synonym.split()) > 2:
                continue

            # ❌ استبعاد الكلمات النادرة جدًا (اختياري بسيط)
            if synonym.isalpha():
                synonyms.add(synonym)

    return list(synonyms)


# In[26]:


print(get_synonyms("machine"))
print(get_synonyms("car"))
print(get_synonyms("learning"))


# In[27]:


def get_synonyms(word):

    synonyms = set()

    synsets = wordnet.synsets(word)

    if not synsets:
        return []

    syn = synsets[0]

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


# In[28]:


print(get_synonyms("machine"))
print(get_synonyms("car"))
print(get_synonyms("learning"))


# In[29]:


def get_synonyms(word):

    synonyms = set()

    synsets = wordnet.synsets(word)[:2]  # 👈 توازن مهم

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


# In[30]:


print(get_synonyms("machine"))
print(get_synonyms("learning"))


# In[31]:


import re

def process_query_service(query, expand=True):

    query_clean = re.sub(r'[^a-z\s]', '', query.lower())
    words = query_clean.split()

    if not expand:
        return " ".join(words)

    local_syns = {
        "reduce": ["lose", "decrease"],
        "weight": ["fat", "body"],
        "internet": ["net", "web"],
        "invest": ["money", "market"]
    }

    expanded_words = list(words)

    for w in words:
        if w in local_syns:
            for syn in local_syns[w]:
                if syn not in expanded_words:
                    expanded_words.append(syn)

    return " ".join(expanded_words)


# In[32]:


query = "reduce weight safely"

result = process_query_service(query)

print("Original:", query)
print("Expanded:", result)


# In[33]:


query = "how to reduce weight safely"
print("Original:", query)
print("Expanded:", process_query_service(query))


# In[ ]:




