import pandas as pd
from collections import defaultdict

from search_core_service import (
    build_inverted_index_and_stats,
    prepare_bm25_weights,
    bm25_search_engine
)

from BERT_Embedding_Service import (
    load_model,
    load_embeddings,
    semantic_search
)

# ==================================================
# Load Documents
# ==================================================
print("Loading processed documents...")

df = pd.read_pickle("processed_documents.pkl")

print(f"Dataset Shape: {df.shape}")

# ==================================================
# Build BM25 Structures
# ==================================================
print("Building BM25 structures...")

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
# Load BERT Model
# ==================================================
print("Loading BERT model...")

model = load_model()

print("Loading embeddings...")

embeddings = load_embeddings(
    "medical_embeddings.npy"
)

print("Embeddings Loaded")

# ==================================================
# Prepare Data
# ==================================================
doc_ids = df["doc_id"].tolist()

texts = (
    df["cleaned_text"]
    .astype(str)
    .tolist()
)

print("Semantic Search Ready")

print("Doc IDs:", len(doc_ids))
print("Texts:", len(texts))

# ==================================================
# Reciprocal Rank Fusion
# ==================================================
def reciprocal_rank_fusion(
    bm25_results,
    bert_results,
    k=60
):
    fused_scores = defaultdict(float)

    # BM25 Ranking
    for rank, (doc_id, score) in enumerate(
        bm25_results,
        start=1
    ):
        fused_scores[doc_id] += (
            1 / (k + rank)
        )

    # BERT Ranking
    for rank, result in enumerate(
        bert_results,
        start=1
    ):
        doc_text = result["text"]

        matched = df[
            df["cleaned_text"] == doc_text
        ]

        if matched.empty:
            continue

        doc_id = matched.iloc[0]["doc_id"]

        fused_scores[doc_id] += (
            1 / (k + rank)
        )

    return sorted(
        fused_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

# ==================================================
# Test Search
# ==================================================
if __name__ == "__main__":

    query = "diabetes treatment"

    print(f"\nQuery: {query}")
    print("=" * 60)

    # ---------------------------
    # BM25 Search
    # ---------------------------
    bm25_results = bm25_search_engine(
        query=query,
        inverted_index=inverted_index,
        bm25_idf=bm25_idf,
        avg_doc_length=avg_doc_length,
        doc_lengths=doc_lengths,
        doc_term_freqs=doc_term_freqs
    )

    print(
        f"BM25 Retrieved: {len(bm25_results)} documents"
    )

    # ---------------------------
    # BERT Search
    # ---------------------------
    bert_results = semantic_search(
        query=query,
        model=model,
        embeddings=embeddings,
        texts=texts,
        doc_ids=doc_ids,
        top_k=100
    )

    print(
        f"BERT Retrieved: {len(bert_results)} documents"
    )

    # ---------------------------
    # Hybrid Search (RRF)
    # ---------------------------
    hybrid_results = reciprocal_rank_fusion(
        bm25_results[:100],
        bert_results
    )

    print(
        f"Hybrid Results: {len(hybrid_results)} documents"
    )

    # ---------------------------
    # Top 10 Rankings
    # ---------------------------
    print("\nTop 10 Results")
    print("=" * 60)

    for rank, (doc_id, score) in enumerate(
        hybrid_results[:10],
        start=1
    ):
        print(
            f"Rank {rank} | "
            f"Doc ID: {doc_id} | "
            f"RRF Score: {score:.6f}"
        )

    # ---------------------------
    # Detailed Top 5 Results
    # ---------------------------
    print("\nDetailed Top 5 Results")
    print("=" * 60)

    for rank, (doc_id, score) in enumerate(
        hybrid_results[:5],
        start=1
    ):
        row = df[
            df["doc_id"] == doc_id
        ].iloc[0]

        print(f"\nRank {rank}")
        print(f"Doc ID: {doc_id}")
        print(f"RRF Score: {score:.6f}")

        print("\nDocument Preview:")
        print(row["cleaned_text"][:300])

        print("-" * 60)