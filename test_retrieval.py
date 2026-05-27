#!/usr/bin/env python
"""Test retrieval with HR-matching queries."""
import sys
sys.path.insert(0, '.')

from app.retrieval.search import detect_mode, fts_search, vector_search, hybrid_search

def query_credit_document(query: str, k: int = 5) -> list[dict]:
    mode = detect_mode(query)
    if mode == "keyword":
        return fts_search(query, k=k)
    if mode == "hybrid":
        return hybrid_search(query, k=k)
    return vector_search(query, k=k)

if __name__ == "__main__":
    test_queries = [
        "How do I apply for leave?",
        "leave",
        "apply leave management system",
    ]

    print("\nRETRIEVAL TEST WITH HR DOCUMENTS\n" + "="*60)
    for q in test_queries:
        print(f"\nQuery : {q}")
        print(f"Mode  : {detect_mode(q)}")
        results = query_credit_document(q, k=3)
        print(f"Hits  : {len(results)}")
        for i, item in enumerate(results, 1):
            meta = item.get("metadata", {})
            print(f"  [{i}] Page {meta.get('page')} — {item['content'][:150]}")
    print("\n" + "="*60)
