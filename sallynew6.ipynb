{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "af07e9e4-b8b7-45d2-af8d-9ddfbc325d4a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "تم تحميل البيانات بنجاح!\n",
      "        doc_id                                      original_text  \\\n",
      "0  NCT00000102  Title: Congenital Adrenal Hyperplasia: Calcium...   \n",
      "1  NCT00000104  Title: Does Lead Burden Alter Neuropsychologic...   \n",
      "2  NCT00000105  Title: Vaccination With Tetanus and KLH to Ass...   \n",
      "3  NCT00000106  Title: 41.8 Degree Centigrade Whole Body Hyper...   \n",
      "4  NCT00000107  Title: Body Water Content in Cyanotic Congenit...   \n",
      "\n",
      "                                        cleaned_text  \n",
      "0  titl congenit adren hyperplasia calcium channe...  \n",
      "1  titl lead burden alter neuropsycholog develop ...  \n",
      "2  titl vaccin tetanu klh assess immun respons co...  \n",
      "3  titl degre centigrad whole bodi hyperthermia t...  \n",
      "4  titl bodi water content cyanot congenit heart ...  \n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "\n",
    "# لاحظي حرف r قبل المسار\n",
    "file_path = r\"C:\\Users\\ev\\Documents\\ir_project-main\\processed_documents.pkl\"\n",
    "df = pd.read_pickle(file_path)\n",
    "\n",
    "# التأكد من نجاح التحميل\n",
    "print(\"تم تحميل البيانات بنجاح!\")\n",
    "print(df.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "1533d822-9970-4a44-a7a0-26f1ca054e5c",
   "metadata": {},
   "outputs": [],
   "source": [
    "import re\n",
    "import nltk\n",
    "\n",
    "from nltk.corpus import stopwords\n",
    "from nltk.tokenize import word_tokenize\n",
    "from nltk.stem import PorterStemmer\n",
    "\n",
    "stop_words = set(stopwords.words(\"english\"))\n",
    "\n",
    "stemmer = PorterStemmer()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "c6b6842b-c8c9-49c4-b3b3-2a939cef07d1",
   "metadata": {},
   "outputs": [],
   "source": [
    "def preprocess_query(query):\n",
    "\n",
    "    query = query.lower()\n",
    "\n",
    "    query = re.sub(\n",
    "        r'[^a-zA-Z\\s]',\n",
    "        '',\n",
    "        query\n",
    "    )\n",
    "\n",
    "    tokens = word_tokenize(query)\n",
    "\n",
    "    tokens = [\n",
    "        word\n",
    "        for word in tokens\n",
    "        if word not in stop_words\n",
    "    ]\n",
    "\n",
    "    tokens = [\n",
    "        stemmer.stem(word)\n",
    "        for word in tokens\n",
    "    ]\n",
    "\n",
    "    return \" \".join(tokens)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "6b951dc8-8b05-4d44-a1a0-538574afb0e9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "treatment heart diseas\n"
     ]
    }
   ],
   "source": [
    "query = \"Treatment for heart disease\"\n",
    "\n",
    "print(\n",
    "    preprocess_query(query)\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "a5dcf899-0438-4a24-b174-809d95b355af",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    titl congenit adren hyperplasia calcium channe...\n",
       "1    titl lead burden alter neuropsycholog develop ...\n",
       "2    titl vaccin tetanu klh assess immun respons co...\n",
       "3    titl degre centigrad whole bodi hyperthermia t...\n",
       "4    titl bodi water content cyanot congenit heart ...\n",
       "Name: cleaned_text, dtype: object"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['cleaned_text'].head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "9f389953-f624-4e3d-a845-96e6993c97c8",
   "metadata": {},
   "outputs": [],
   "source": [
    "queries = [\n",
    "    \"Heart disease treatment\",\n",
    "    \"Breast cancer therapy\",\n",
    "    \"Diabetes clinical trial\",\n",
    "    \"Congenital heart disease\",\n",
    "    \"Calcium channel blockers\",\n",
    "    \"Body water content\",\n",
    "    \"Neuropsychological development\",\n",
    "    \"Immune response vaccination\",\n",
    "    \"Whole body hyperthermia\",\n",
    "    \"Lead burden effects\"\n",
    "]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "9cf9cce2-cd41-4f73-a37b-d25afe8e66e6",
   "metadata": {},
   "outputs": [],
   "source": [
    "def query_service(query):\n",
    "\n",
    "    processed_query = process_query(query)\n",
    "\n",
    "    return {\n",
    "        \"original_query\": query,\n",
    "        \"processed_query\": processed_query\n",
    "    }"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "385ef211-6a4e-4bab-b3ac-f77415812a57",
   "metadata": {},
   "outputs": [],
   "source": [
    "import re\n",
    "import nltk\n",
    "\n",
    "from nltk.corpus import stopwords\n",
    "from nltk.tokenize import word_tokenize\n",
    "from nltk.stem import PorterStemmer\n",
    "\n",
    "stop_words = set(stopwords.words(\"english\"))\n",
    "stemmer = PorterStemmer()\n",
    "\n",
    "def process_query(query):\n",
    "\n",
    "    query = query.lower()\n",
    "\n",
    "    query = re.sub(r'[^a-zA-Z\\s]', '', query)\n",
    "\n",
    "    tokens = word_tokenize(query)\n",
    "\n",
    "    tokens = [\n",
    "        word for word in tokens\n",
    "        if word not in stop_words\n",
    "    ]\n",
    "\n",
    "    tokens = [\n",
    "        stemmer.stem(word)\n",
    "        for word in tokens\n",
    "    ]\n",
    "\n",
    "    return \" \".join(tokens)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "11a97d41-1e12-4823-a533-d4464cd6767d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{'original_query': 'Heart disease treatment', 'processed_query': 'heart diseas treatment'}\n",
      "--------------------------------------------------\n",
      "{'original_query': 'Breast cancer therapy', 'processed_query': 'breast cancer therapi'}\n",
      "--------------------------------------------------\n",
      "{'original_query': 'Diabetes clinical trial', 'processed_query': 'diabet clinic trial'}\n",
      "--------------------------------------------------\n",
      "{'original_query': 'Congenital heart disease', 'processed_query': 'congenit heart diseas'}\n",
      "--------------------------------------------------\n",
      "{'original_query': 'Calcium channel blockers', 'processed_query': 'calcium channel blocker'}\n",
      "--------------------------------------------------\n",
      "{'original_query': 'Body water content', 'processed_query': 'bodi water content'}\n",
      "--------------------------------------------------\n",
      "{'original_query': 'Neuropsychological development', 'processed_query': 'neuropsycholog develop'}\n",
      "--------------------------------------------------\n",
      "{'original_query': 'Immune response vaccination', 'processed_query': 'immun respons vaccin'}\n",
      "--------------------------------------------------\n",
      "{'original_query': 'Whole body hyperthermia', 'processed_query': 'whole bodi hyperthermia'}\n",
      "--------------------------------------------------\n",
      "{'original_query': 'Lead burden effects', 'processed_query': 'lead burden effect'}\n",
      "--------------------------------------------------\n"
     ]
    }
   ],
   "source": [
    "for q in queries:\n",
    "    print(query_service(q))\n",
    "    print(\"-\" * 50)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "f2214020-bbff-47d5-b554-16b9c83a578b",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package wordnet to\n",
      "[nltk_data]     C:\\Users\\ev\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package wordnet is already up-to-date!\n"
     ]
    }
   ],
   "source": [
    "import nltk\n",
    "nltk.download('wordnet')\n",
    "\n",
    "from nltk.corpus import wordnet"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "b2ee181a-df3b-453b-8e3e-7e919c78a103",
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_synonyms(word):\n",
    "\n",
    "    synonyms = set()\n",
    "\n",
    "    for syn in wordnet.synsets(word):\n",
    "        for lemma in syn.lemmas():\n",
    "            synonyms.add(\n",
    "                lemma.name().replace(\"_\",\" \")\n",
    "            )\n",
    "\n",
    "    return list(synonyms)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "c72460dc-8854-41e4-9916-694d7317a9ee",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['inwardness', 'ticker', 'nerve', 'heart', 'bosom', 'sum', 'tenderness', 'center', 'heart and soul', 'warmness']\n",
      "['disease']\n"
     ]
    }
   ],
   "source": [
    "print(get_synonyms(\"heart\")[:10])\n",
    "print(get_synonyms(\"disease\")[:10])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "0b8b9ffb-f4a9-4d2e-a729-b40d7c84a351",
   "metadata": {},
   "outputs": [],
   "source": [
    "def expand_query(query):\n",
    "\n",
    "    words = query.lower().split()\n",
    "\n",
    "    expanded = words.copy()\n",
    "\n",
    "    for word in words:\n",
    "        expanded.extend(\n",
    "            get_synonyms(word)[:2]\n",
    "        )\n",
    "\n",
    "    return \" \".join(set(expanded))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "c296d58b-400f-49ca-83c6-d7d4eace0377",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "inwardness disease heart ticker\n"
     ]
    }
   ],
   "source": [
    "print(\n",
    "    expand_query(\n",
    "        \"heart disease\"\n",
    "    )\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "6190a27b-735d-47f3-a138-f911c694e934",
   "metadata": {},
   "outputs": [],
   "source": [
    "from textblob import TextBlob\n",
    "\n",
    "def suggest_query(query):\n",
    "\n",
    "    return str(\n",
    "        TextBlob(query).correct()\n",
    "    )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "4bde7d30-0118-44a0-bc35-78580cbde2f4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "hart disease\n"
     ]
    }
   ],
   "source": [
    "print(\n",
    "    suggest_query(\n",
    "        \"hart diseas\"\n",
    "    )\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "4649b7db-a833-4c83-afe4-09509ab0bfe8",
   "metadata": {},
   "outputs": [],
   "source": [
    "def query_service_v3(query):\n",
    "\n",
    "    corrected_query = str(TextBlob(query).correct())\n",
    "\n",
    "    processed_query = process_query(corrected_query)\n",
    "\n",
    "    expanded_query = expand_query(corrected_query)\n",
    "\n",
    "    return {\n",
    "        \"original_query\": query,\n",
    "        \"corrected_query\": corrected_query,\n",
    "        \"processed_query\": processed_query,\n",
    "        \"expanded_query\": expanded_query\n",
    "    }"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "c7ffc148-94c9-4056-9697-84f3c3187e88",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{'original_query': 'hart diseas treatment', 'corrected_query': 'hart disease treatment', 'processed_query': 'hart diseas treatment', 'expanded_query': 'disease handling Lorenz Milton Hart intervention stag hart treatment'}\n"
     ]
    }
   ],
   "source": [
    "result = query_service_v3(\"hart diseas treatment\")\n",
    "print(result)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "3bbac0ac-6752-4018-a045-2cb00ab210a3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{'original_query': 'heart disease treatment', 'corrected_query': 'heart disease treatment', 'processed_query': 'heart diseas treatment', 'expanded_query': 'inwardness disease ticker handling heart intervention treatment'}\n"
     ]
    }
   ],
   "source": [
    "print(query_service_v3(\"heart disease treatment\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "fee7dd05-d01e-46fb-8287-315e1d262153",
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_synonyms(word):\n",
    "\n",
    "    synonyms = set()\n",
    "\n",
    "    for syn in wordnet.synsets(word):\n",
    "\n",
    "        for lemma in syn.lemmas():\n",
    "            synonym = lemma.name().replace(\"_\", \" \")\n",
    "\n",
    "            if synonym.lower() != word.lower():\n",
    "                synonyms.add(synonym.lower())\n",
    "\n",
    "        break\n",
    "\n",
    "    return list(synonyms)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "ca9c8d2b-05b8-460f-b133-f3c99c3812dc",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['bosom']\n",
      "['malignant neoplastic disease']\n",
      "['intervention']\n"
     ]
    }
   ],
   "source": [
    "print(get_synonyms(\"heart\"))\n",
    "print(get_synonyms(\"cancer\"))\n",
    "print(get_synonyms(\"treatment\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "18db41fe-b08d-4d53-bb3b-332ad9f07416",
   "metadata": {},
   "outputs": [],
   "source": [
    "medical_synonyms = {\n",
    "    \"heart\": [\"cardiac\"],\n",
    "    \"cancer\": [\"tumor\", \"carcinoma\"],\n",
    "    \"disease\": [\"illness\", \"disorder\"],\n",
    "    \"treatment\": [\"therapy\"],\n",
    "    \"vaccination\": [\"immunization\"]\n",
    "}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "49158a79-b682-4e5a-9fdb-29c422bfe9ae",
   "metadata": {},
   "outputs": [],
   "source": [
    "def expand_query(query):\n",
    "\n",
    "    processed_query = process_query(query)\n",
    "    words = processed_query.split()\n",
    "\n",
    "    expanded = set(words)\n",
    "\n",
    "    for w in words:\n",
    "        if w in medical_synonyms:\n",
    "            expanded.update(medical_synonyms[w])\n",
    "\n",
    "    return list(expanded)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "f1cf659b-8595-44af-809c-d23c7c8af8a1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['pump', 'ticker', 'spunk', 'nerve', 'bosom', 'mettle']\n",
      "[]\n",
      "['malignant neoplastic disease', 'crab']\n"
     ]
    }
   ],
   "source": [
    "print(get_synonyms(\"heart\"))\n",
    "print(get_synonyms(\"disease\"))\n",
    "print(get_synonyms(\"cancer\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "957bf7bc-de2e-4fd8-82d7-e8ec363b24f9",
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_synonyms(word):\n",
    "\n",
    "    synonyms = set()\n",
    "\n",
    "    synsets = wordnet.synsets(word)[:5]\n",
    "\n",
    "    for syn in synsets:\n",
    "        for lemma in syn.lemmas():\n",
    "\n",
    "            synonym = lemma.name().replace(\"_\", \" \").lower()\n",
    "\n",
    "            if synonym == word.lower():\n",
    "                continue\n",
    "\n",
    "            if len(synonym.split()) > 2:\n",
    "                continue\n",
    "\n",
    "            if synonym.isalpha():\n",
    "                synonyms.add(synonym)\n",
    "\n",
    "    return list(synonyms)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "65c2a047-36f0-4641-aaaa-5a505753b548",
   "metadata": {},
   "outputs": [],
   "source": [
    "medical_synonyms = {\n",
    "    \"heart\": [\"cardiac\"],\n",
    "    \"cancer\": [\"tumor\"],\n",
    "    \"treatment\": [\"therapy\"],\n",
    "    \"vaccination\": [\"immunization\"],\n",
    "    \"disease\": [\"illness\"]\n",
    "}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "10b3d6f8-f829-4376-8305-ec7ca343d4bf",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['inwardness', 'ticker', 'nerve', 'bosom', 'sum', 'center', 'centre', 'kernel', 'pump', 'meat', 'eye', 'core', 'substance', 'gist', 'pith', 'spunk', 'middle', 'mettle', 'essence', 'marrow', 'nub']\n",
      "[]\n",
      "['crab']\n",
      "['intervention', 'discussion', 'discourse', 'handling']\n",
      "['inoculation']\n"
     ]
    }
   ],
   "source": [
    "print(get_synonyms(\"heart\"))\n",
    "print(get_synonyms(\"disease\"))\n",
    "print(get_synonyms(\"cancer\"))\n",
    "print(get_synonyms(\"treatment\"))\n",
    "print(get_synonyms(\"vaccination\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "d126ef35-32c5-40a5-bde9-126f644cbac6",
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_synonyms(word):\n",
    "\n",
    "    synonyms = set()\n",
    "\n",
    "    synsets = wordnet.synsets(word)\n",
    "\n",
    "    if not synsets:\n",
    "        return []\n",
    "\n",
    "    syn = synsets[0]   # أول معنى فقط (Improvement مهم)\n",
    "\n",
    "    for lemma in syn.lemmas():\n",
    "\n",
    "        synonym = lemma.name().replace(\"_\", \" \").lower()\n",
    "\n",
    "        if synonym == word.lower():\n",
    "            continue\n",
    "\n",
    "        if not synonym.isalpha():\n",
    "            continue\n",
    "\n",
    "        if len(synonym) < 3:\n",
    "            continue\n",
    "\n",
    "        synonyms.add(synonym)\n",
    "\n",
    "    return list(synonyms)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "c984c8b0-e0e6-4377-8591-aca6febe54a7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['bosom']\n",
      "[]\n",
      "[]\n",
      "['intervention']\n"
     ]
    }
   ],
   "source": [
    "print(get_synonyms(\"heart\"))\n",
    "print(get_synonyms(\"disease\"))\n",
    "print(get_synonyms(\"cancer\"))\n",
    "print(get_synonyms(\"treatment\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "c156cc54-4aef-4252-ac01-6df818cc48bc",
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_synonyms(word):\n",
    "\n",
    "    synonyms = set()\n",
    "\n",
    "    synsets = wordnet.synsets(word)[:2]\n",
    "\n",
    "    for syn in synsets:\n",
    "\n",
    "        for lemma in syn.lemmas():\n",
    "\n",
    "            synonym = lemma.name().replace(\"_\", \" \").lower()\n",
    "\n",
    "            if synonym == word.lower():\n",
    "                continue\n",
    "\n",
    "            if not synonym.isalpha():\n",
    "                continue\n",
    "\n",
    "            if len(synonym) < 3:\n",
    "                continue\n",
    "\n",
    "            synonyms.add(synonym)\n",
    "\n",
    "    return list(synonyms)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "c11ee023-c425-49d2-adfe-5183dc3774ed",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['pump', 'ticker', 'bosom']\n",
      "[]\n",
      "['crab']\n",
      "['intervention', 'handling']\n"
     ]
    }
   ],
   "source": [
    "print(get_synonyms(\"heart\"))\n",
    "print(get_synonyms(\"disease\"))\n",
    "print(get_synonyms(\"cancer\"))\n",
    "print(get_synonyms(\"treatment\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "a54905f1-0dd5-4c5c-9d63-6376e7c1b0fd",
   "metadata": {},
   "outputs": [],
   "source": [
    "import re\n",
    "\n",
    "def process_query_service(query, expand=True):\n",
    "\n",
    "    query_clean = re.sub(r'[^a-z\\s]', '', query.lower())\n",
    "    words = query_clean.split()\n",
    "\n",
    "    if not expand:\n",
    "        return \" \".join(words)\n",
    "\n",
    "    medical_syns = {\n",
    "        \"heart\": [\"cardiac\"],\n",
    "        \"disease\": [\"illness\", \"disorder\"],\n",
    "        \"cancer\": [\"tumor\", \"carcinoma\"],\n",
    "        \"diabetes\": [\"diabetic\"],\n",
    "        \"treatment\": [\"therapy\"],\n",
    "        \"vaccination\": [\"immunization\"],\n",
    "        \"immune\": [\"immunity\"],\n",
    "        \"clinical\": [\"medical\"],\n",
    "        \"trial\": [\"study\"]\n",
    "    }\n",
    "\n",
    "    expanded_words = list(words)\n",
    "\n",
    "    for w in words:\n",
    "        if w in medical_syns:\n",
    "            for syn in medical_syns[w]:\n",
    "                if syn not in expanded_words:\n",
    "                    expanded_words.append(syn)\n",
    "\n",
    "    return \" \".join(expanded_words)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "4f156a5d-4217-4b89-80a2-e4b8b955e484",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "heart disease treatment cardiac illness disorder therapy\n",
      "breast cancer therapy tumor carcinoma\n",
      "diabetes clinical trial diabetic medical study\n",
      "immune response vaccination immunity immunization\n"
     ]
    }
   ],
   "source": [
    "print(process_query_service(\"heart disease treatment\"))\n",
    "print(process_query_service(\"breast cancer therapy\"))\n",
    "print(process_query_service(\"diabetes clinical trial\"))\n",
    "print(process_query_service(\"immune response vaccination\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "aab56e08-01fc-4025-9201-400e45b78763",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[NbConvertApp] Converting notebook sallynew.ipynb to script\n",
      "[NbConvertApp] Writing 7753 bytes to sallynew.py\n"
     ]
    }
   ],
   "source": [
    "!jupyter nbconvert --to script sallynew.ipynb"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "efac72ca-45cd-42a3-8eac-7b8042a86f4c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "تم العثور على الملف وتحميله بنجاح!\n",
      "        doc_id                                      original_text  \\\n",
      "0  NCT00000102  Title: Congenital Adrenal Hyperplasia: Calcium...   \n",
      "1  NCT00000104  Title: Does Lead Burden Alter Neuropsychologic...   \n",
      "2  NCT00000105  Title: Vaccination With Tetanus and KLH to Ass...   \n",
      "3  NCT00000106  Title: 41.8 Degree Centigrade Whole Body Hyper...   \n",
      "4  NCT00000107  Title: Body Water Content in Cyanotic Congenit...   \n",
      "\n",
      "                                        cleaned_text  \n",
      "0  titl congenit adren hyperplasia calcium channe...  \n",
      "1  titl lead burden alter neuropsycholog develop ...  \n",
      "2  titl vaccin tetanu klh assess immun respons co...  \n",
      "3  titl degre centigrad whole bodi hyperthermia t...  \n",
      "4  titl bodi water content cyanot congenit heart ...  \n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import os\n",
    "\n",
    "# ضعي المسار الكامل هنا (تأكدي من وجود حرف r قبل المسار)\n",
    "file_path = r\"C:\\Users\\ev\\Documents\\ir_project-main\\processed_documents.pkl\"\n",
    "\n",
    "# التحقق من وجود الملف قبل محاولة فتحه\n",
    "if os.path.exists(file_path):\n",
    "    df = pd.read_pickle(file_path)\n",
    "    print(\"تم العثور على الملف وتحميله بنجاح!\")\n",
    "    print(df.head())\n",
    "else:\n",
    "    print(f\"خطأ: الملف غير موجود في المسار: {file_path}\")\n",
    "    print(\"يرجى التأكد من أن الملف موجود في هذا المجلد بالتحديد.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "14f8cde9-a364-4c9d-a011-ba2ccbc6195b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "        doc_id                                      original_text  \\\n",
      "0  NCT00000102  Title: Congenital Adrenal Hyperplasia: Calcium...   \n",
      "1  NCT00000104  Title: Does Lead Burden Alter Neuropsychologic...   \n",
      "2  NCT00000105  Title: Vaccination With Tetanus and KLH to Ass...   \n",
      "3  NCT00000106  Title: 41.8 Degree Centigrade Whole Body Hyper...   \n",
      "4  NCT00000107  Title: Body Water Content in Cyanotic Congenit...   \n",
      "\n",
      "                                        cleaned_text  \n",
      "0  titl congenit adren hyperplasia calcium channe...  \n",
      "1  titl lead burden alter neuropsycholog develop ...  \n",
      "2  titl vaccin tetanu klh assess immun respons co...  \n",
      "3  titl degre centigrad whole bodi hyperthermia t...  \n",
      "4  titl bodi water content cyanot congenit heart ...  \n"
     ]
    }
   ],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "57a418f3-5c2d-4008-b4be-888e93333c25",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "تم حفظ الملف بنجاح في المسار: C:\\Users\\ev\\Documents\\ir_project-main\\processed_documents.pkl\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import os\n",
    "\n",
    "# 1. تحديد المسار بدقة باستخدام الـ raw string (r) لتجنب مشاكل الـ backslashes\n",
    "target_dir = r\"C:\\Users\\ev\\Documents\\ir_project-main\"\n",
    "file_name = \"processed_documents.pkl\"\n",
    "\n",
    "# 2. دمج المسار والاسم برمجياً\n",
    "full_path = os.path.join(target_dir, file_name)\n",
    "\n",
    "# 3. التأكد من أن المجلد موجود، وإن لم يكن كذلك فسيقوم بإنشائه\n",
    "if not os.path.exists(target_dir):\n",
    "    os.makedirs(target_dir)\n",
    "\n",
    "# 4. حفظ ملف الـ Pickle في المسار المحدد\n",
    "df.to_pickle(full_path)\n",
    "\n",
    "print(f\"تم حفظ الملف بنجاح في المسار: {full_path}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "77af2c41-c772-49ca-8bfd-17401f0cc758",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "جاري البحث... قد يستغرق الأمر ثوانٍ قليلة.\n",
      "وجدته! الملف موجود في: C:\\Users\\ev\\Documents\\ir_project-main\\processed_documents.pkl\n"
     ]
    }
   ],
   "source": [
    "import os\n",
    "\n",
    "# ابحثي في المجلد الرئيسي للمستخدم\n",
    "search_path = r\"C:\\Users\\ev\"\n",
    "target_file = \"processed_documents.pkl\"\n",
    "\n",
    "print(\"جاري البحث... قد يستغرق الأمر ثوانٍ قليلة.\")\n",
    "\n",
    "for root, dirs, files in os.walk(search_path):\n",
    "    if target_file in files:\n",
    "        print(f\"وجدته! الملف موجود في: {os.path.join(root, target_file)}\")\n",
    "        break\n",
    "else:\n",
    "    print(\"عذراً، لم أجد الملف في مجلد المستخدم. هل أنتِ متأكدة من كتابة الاسم بشكل صحيح؟\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "68480ed6-e9d4-4451-93c6-dee845d728a2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "محتويات المجلد C:\\Users\\ev\\Documents\\ir_project-main:\n",
      " - .git\n",
      " - .ipynb_checkpoints\n",
      " - anaconda_projects\n",
      " - app.py\n",
      " - Document_Clustering_Service.ipynb\n",
      " - embedding_service.py\n",
      " - ghazal.ipynb\n",
      " - ghazalsally.ipynb\n",
      " - ghazalsally.py\n",
      " - git\n",
      " - lama update 5.py\n",
      " - Lama5.ipynb\n",
      " - lama8+.ipynb\n",
      " - lama8+.py\n",
      " - lama_evaluation_rag_service.py\n",
      " - ltr_service.ipynb\n",
      " - ltr_service.py\n",
      " - ltr_service_bak.ipynb\n",
      " - maya.ipynb\n",
      " - my_old_files\n",
      " - new_maria.ipynb\n",
      " - parallel_hybrid_search.py\n",
      " - processed_documents.pkl\n",
      " - quora_cleaned_sample.csv\n",
      " - quora_embeddings.npy\n",
      " - retrieval_real_evaluation_report.csv\n",
      " - retrieval_service.py\n",
      " - sally3.ipynb\n",
      " - sally3.py\n",
      " - Untitled1.ipynb\n",
      " - Untitled14.ipynb\n",
      " - Untitled15.ipynb\n",
      " - Untitled19.ipynb\n",
      " - Untitled3.ipynb\n",
      " - __pycache__\n",
      "\n",
      "✅ تم العثور على الملف: processed_documents.pkl موجود داخل المجلد!\n"
     ]
    }
   ],
   "source": [
    "import os\n",
    "\n",
    "target_dir = r\"C:\\Users\\ev\\Documents\\ir_project-main\"\n",
    "\n",
    "# التأكد من أن المجلد موجود\n",
    "if os.path.exists(target_dir):\n",
    "    files = os.listdir(target_dir)\n",
    "    print(f\"محتويات المجلد {target_dir}:\")\n",
    "    for f in files:\n",
    "        print(f\" - {f}\")\n",
    "        \n",
    "    if \"processed_documents.pkl\" in files:\n",
    "        print(\"\\n✅ تم العثور على الملف: processed_documents.pkl موجود داخل المجلد!\")\n",
    "    else:\n",
    "        print(\"\\n❌ الملف غير موجود في هذا المجلد. هل هو باسم مختلف؟\")\n",
    "else:\n",
    "    print(f\"المجلد {target_dir} غير موجود. تأكدي من المسار.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "d35bc667-6d0b-4ddc-b846-a0dc7f4a557a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ تم نقل الملف بنجاح إلى: C:\\Users\\ev\\Desktop\\New folder\\processed_documents.pkl\n"
     ]
    }
   ],
   "source": [
    "import shutil\n",
    "import os\n",
    "\n",
    "# 1. المسار الحالي للملف (حيث وجدناه سابقاً)\n",
    "source_path = r\"C:\\Users\\ev\\Documents\\ir_project-main\\processed_documents.pkl\"\n",
    "\n",
    "# 2. مسار الوجهة الجديد\n",
    "target_folder = r\"C:\\Users\\ev\\Desktop\\New folder\"\n",
    "target_path = os.path.join(target_folder, \"processed_documents.pkl\")\n",
    "\n",
    "# 3. التأكد من وجود المجلد الوجهة، وإن لم يكن موجوداً نقوم بإنشائه\n",
    "if not os.path.exists(target_folder):\n",
    "    os.makedirs(target_folder)\n",
    "    print(f\"تم إنشاء المجلد: {target_folder}\")\n",
    "\n",
    "# 4. عملية النقل\n",
    "try:\n",
    "    shutil.move(source_path, target_path)\n",
    "    print(f\"✅ تم نقل الملف بنجاح إلى: {target_path}\")\n",
    "except Exception as e:\n",
    "    print(f\"حدث خطأ أثناء النقل: {e}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "a998f0c2-4762-49d3-b354-b08b22dcf978",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "المجلد الذي يعمل فيه Jupyter الآن هو: C:\\Users\\ev\\anaconda_projects\\dd4c724e-f9ab-4a42-a7b6-82a4164cdb92\n"
     ]
    }
   ],
   "source": [
    "import os\n",
    "print(\"المجلد الذي يعمل فيه Jupyter الآن هو:\", os.getcwd())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "8f8aad3f-f8c4-4684-bc84-a7b6ad622dd9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "تم حفظ الملف بنجاح في: C:\\Users\\ev\\Documents\\ir_project-main\\processed_documents.pkl\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import os\n",
    "\n",
    "# المسار الدقيق كما يظهر في جهازك\n",
    "target_path = r\"C:\\Users\\ev\\Documents\\ir_project-main\\processed_documents.pkl\"\n",
    "\n",
    "# حفظ الـ DataFrame\n",
    "df.to_pickle(target_path)\n",
    "\n",
    "print(f\"تم حفظ الملف بنجاح في: {target_path}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a486981a-70f4-4eff-89b8-f0bccd2a06c7",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
