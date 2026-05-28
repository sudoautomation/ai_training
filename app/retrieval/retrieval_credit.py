from app.retrieval.search import detect_mode, fts_search, hybrid_search, vector_search


def query_credit_document(query: str, k: int = 5) -> list[dict]:
    """Public entry point for Credit Risk FAQ retrieval.

    Automatically selects the best retrieval mode:
      - 'keyword' → RBI codes, acronyms, numeric IDs       → fts_search
      - 'hybrid'  → short natural-language queries         → hybrid_search (RRF)
      - 'vector'  → full conversational / semantic queries → vector_search

    Args:
        query: Natural language or keyword query string.
        k:     Number of results to return (default 5).

    Returns:
        List of dicts with keys: content, metadata (+ fts_rank for keyword mode).
    """
    mode = detect_mode(query)

    if mode == "keyword":
        return fts_search(query, k=k)

    if mode == "hybrid":
        return hybrid_search(query, k=k)

    return vector_search(query, k=k)


# ── Quick smoke test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    from app.retrieval.search import detect_mode

    test_queries = [
        "what is the DTI threshold for personal loans?",
        "LTV",
        "NPA classification borrower defaults",
    ]

    for q in test_queries:
        print(f"\nQuery : {q}")
        print(f"Mode  : {detect_mode(q)}")
        results = query_credit_document(q, k=7)
        print(f"Hits  : {len(results)}")
        for i, item in enumerate(results, 1):
            meta = item.get("metadata", {})
            print(f"  [{i}] Page {meta.get('page')} — {item['content'][:200]}")