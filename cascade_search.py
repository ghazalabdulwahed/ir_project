import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from search_core_service import (
    build_inverted_index_and_stats,
    prepare_bm25_weights,
    bm25_search_engine
)

from BERT_Embedding_Service import (
    load_model,
    load_embeddings
)


# ==================================================
# Load Processed Documents
# ==================================================
print("Loading processed documents...")

df = pd.read_pickle("processed_documents.pkl")

print(f"Dataset Shape: {df.shape}")


# ==================================================
# Build BM25 Structures
# ==================================================
print("Building inverted index...")

inverted_index, doc_lengths, doc_term_freqs = (
    build_inverted_index_and_stats(df)
)

bm25_idf = prepare_bm25_weights(
    df,
    inverted_index
)

avg_doc_length = (
    sum(doc_lengths.values())
    / len(doc_lengths)
)

print("BM25 Ready")


# ==================================================
# Load BERT Model and Embeddings
# ==================================================
print("Loading BERT model...")

model = load_model()

print("Loading embeddings...")

embeddings = load_embeddings(
    "medical_embeddings.npy"
)

print("Embeddings Loaded")


# ==================================================
# Create Doc ID → Row Mapping
# ==================================================
docid_to_row = {}

for idx, doc_id in enumerate(df["doc_id"]):
    docid_to_row[doc_id] = idx

print("Mapping Ready")


# ==================================================
# Serial Cascade Search
# ==================================================
def serial_cascade_search(
    query,
    top_bm25=500,
    final_k=10
):
    """
    Stage 1:
        BM25 retrieves top candidates.

    Stage 2:
        BERT reranks BM25 candidates using cosine similarity.
    """

    bm25_results = bm25_search_engine(
        query=query,
        inverted_index=inverted_index,
        bm25_idf=bm25_idf,
        avg_doc_length=avg_doc_length,
        doc_lengths=doc_lengths,
        doc_term_freqs=doc_term_freqs
    )

    candidate_docs = bm25_results[:top_bm25]

    if len(candidate_docs) == 0:
        return []

    query_embedding = model.encode([query])

    reranked = []

    for doc_id, bm25_score in candidate_docs:

        row_idx = docid_to_row[doc_id]

        doc_embedding = embeddings[row_idx].reshape(1, -1)

        semantic_score = cosine_similarity(
            query_embedding,
            doc_embedding
        )[0][0]

        reranked.append(
            (
                doc_id,
                semantic_score
            )
        )

    reranked.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return reranked[:final_k]


# ==================================================
# Test Search
# ==================================================
if __name__ == "__main__":

    query = "diabetes treatment"

    print(f"\nQuery: {query}")
    print("=" * 60)

    cascade_results = serial_cascade_search(
        query=query,
        top_bm25=500,
        final_k=10
    )

    if not cascade_results:
        print("No results found.")

    else:
        for rank, (doc_id, score) in enumerate(
            cascade_results,
            start=1
        ):

            row = df[
                df["doc_id"] == doc_id
            ].iloc[0]

            print(f"\nRank {rank}")
            print(f"Doc ID: {doc_id}")
            print(f"Semantic Score: {score:.4f}")

            print("\nDocument Preview:")
            print(row["cleaned_text"][:300])

            print("-" * 60)