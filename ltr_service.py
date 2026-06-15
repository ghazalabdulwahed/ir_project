def rerank(results):
    return sorted(
        results,
        key=lambda x: x["score"],
        reverse=True
    )