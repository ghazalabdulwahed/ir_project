import os
import pandas as pd
from collections import defaultdict
from sentence_transformers import SentenceTransformer

from retrieval_service import (
    build_inverted_index,
    prepare_bm25,
    bm25_search
)

from embedding_service import SemanticSearchService


# =========================
# 1. Load Dataset
# =========================
df = pd.read_csv("quora_cleaned_sample.csv")


# =========================
# 2. BM25 Setup
# =========================
inverted_index = build_inverted_index(df)

bm25_idf, avg_doc_length = prepare_bm25(
    df,
    inverted_index
)


# =========================
# 3. Check embeddings file
# =========================
print("Embeddings file exists:", os.path.exists("quora_embeddings.npy"))


# =========================
# 4. Load Semantic Service
# =========================
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    local_files_only=True
)

search_service = SemanticSearchService()

search_service.initialize_service(
    load_embeddings=True
)


# =========================
# 5. RRF (Hybrid Fusion)
# =========================
def reciprocal_rank_fusion(
    bm25_results,
    bert_results,
    k=60
):
    fused_scores = defaultdict(float)

    # BM25 results
    for rank, (doc_id, score) in enumerate(bm25_results, start=1):
        fused_scores[doc_id] += 1 / (k + rank)

    # BERT results
    for rank, result in enumerate(bert_results, start=1):
        doc_id = result["doc_id"]
        fused_scores[doc_id] += 1 / (k + rank)

    # Sort final results
    return sorted(
        fused_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )


# =========================
# 6. Run Hybrid Search
# =========================
query = "machine learning"

bm25_results = bm25_search(
    query,
    df,
    inverted_index,
    bm25_idf,
    avg_doc_length
)

bert_results = search_service.search_semantic(
    query,
    top_k=100
)

hybrid_results = reciprocal_rank_fusion(
    bm25_results[:100],
    bert_results,
    k=60
)


# =========================
# 7. Print Top Results
# =========================
print("\n===== TOP 10 HYBRID RESULTS =====\n")

for rank, (doc_id, score) in enumerate(hybrid_results[:10], start=1):
    print(f"Rank {rank} | Doc ID: {doc_id} | Score: {score:.6f}")


# =========================
# 8. Show Detailed Results
# =========================
print("\n===== DETAILED TOP 5 =====\n")

for rank, (doc_id, score) in enumerate(hybrid_results[:5], start=1):
    row = df[df["doc_id"] == doc_id].iloc[0]

    print(f"Rank {rank}")
    print(f"Doc ID: {doc_id}")
    print(f"RRF Score: {score:.6f}")
    print(row["cleaned_text"][:200])
    print("-" * 50)