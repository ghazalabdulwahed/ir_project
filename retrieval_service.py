from collections import defaultdict, Counter
import math
import pandas as pd

#بناء الفهرس المعكوس

def build_inverted_index(df):

    inverted_index = defaultdict(set)

    for _, row in df.iterrows():

        doc_id = row["doc_id"]

        words = str(row["cleaned_text"]).split()

        for word in words:
            inverted_index[word].add(doc_id)

    return inverted_index

#حساب TF-IDF

def build_tfidf(df, inverted_index):

    tf_docs = {}

    for _, row in df.iterrows():

        doc_id = row["doc_id"]

        words = str(row["cleaned_text"]).split()

        counts = Counter(words)

        tf_docs[doc_id] = {
            word: math.log(1 + count)
            for word, count in counts.items()
        }

    N = len(df)

    idf = {
        word: math.log(1 + (N / len(docs)))
        for word, docs in inverted_index.items()
    }

    tfidf_docs = {}

    for doc_id, tf_dict in tf_docs.items():

        tfidf_docs[doc_id] = {
            word: tf_value * idf[word]
            for word, tf_value in tf_dict.items()
        }

    return tfidf_docs, idf

#بناء BM25

def prepare_bm25(df, inverted_index):

    doc_lengths = {}

    for _, row in df.iterrows():

        doc_id = row["doc_id"]

        words = str(row["cleaned_text"]).split()

        doc_lengths[doc_id] = len(words)

    avg_doc_length = (
        sum(doc_lengths.values()) /
        len(doc_lengths)
    )

    N = len(df)

    bm25_idf = {}

    for term, docs in inverted_index.items():

        df_term = len(docs)

        bm25_idf[term] = math.log(
            ((N - df_term + 0.5) /
             (df_term + 0.5)) + 1
        )

    return bm25_idf, avg_doc_length

#البحث باستخدام BM25
def bm25_search(
    query,
    df,
    inverted_index,
    bm25_idf,
    avg_doc_length,
    k1=1.5,
    b=0.75
):

    scores = {}

    query_terms = query.lower().split()

    candidate_docs = set()

    for term in query_terms:

        if term in inverted_index:
            candidate_docs.update(
                inverted_index[term]
            )

    for doc_id in candidate_docs:

        row = df[df["doc_id"] == doc_id].iloc[0]

        words = str(row["cleaned_text"]).split()

        doc_len = len(words)

        term_freqs = Counter(words)

        score = 0

        for term in query_terms:

            if term not in term_freqs:
                continue

            tf = term_freqs[term]

            idf = bm25_idf.get(term, 0)

            numerator = tf * (k1 + 1)

            denominator = (
                tf +
                k1 * (
                    1 - b +
                    b * (doc_len / avg_doc_length)
                )
            )

            score += idf * (
                numerator / denominator
            )

        scores[doc_id] = score

    return sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

#اختبار الخدمة

if __name__ == "__main__":
    

    df = pd.read_csv(
        "quora_cleaned_sample.csv"
    )

    inverted_index = build_inverted_index(df)

    tfidf_docs, idf = build_tfidf(
        df,
        inverted_index
    )

    bm25_idf, avg_doc_length = prepare_bm25(
        df,
        inverted_index
    )

    results = bm25_search(
        "machine learning",
        df,
        inverted_index,
        bm25_idf,
        avg_doc_length
    )

    print("Service Running\n", results[:5])
    

