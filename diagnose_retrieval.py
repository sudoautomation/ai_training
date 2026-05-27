"""Diagnostic test to trace query detection, search mode, and ranking."""

from app.retrieval.search import (
    detect_mode, fts_search, vector_search, hybrid_search, 
    _KEYWORD_RE, _reranker, COLLECTION_NAME, get_vector_store
)

def diagnose_query(query: str, k: int = 5):
    """Trace a single query through the entire pipeline."""
    print(f"\n{'='*70}")
    print(f"QUERY: {query}")
    print(f"{'='*70}")
    
    # 1. Mode detection
    mode = detect_mode(query)
    print(f"\n1️⃣  MODE DETECTION")
    print(f"   Detected mode: {mode}")
    print(f"   Keyword match: {bool(_KEYWORD_RE.search(query))}")
    print(f"   Word count: {len(query.split())}")
    
    # 2. Try each search method
    print(f"\n2️⃣  SEARCH RESULTS BY METHOD")
    
    # FTS
    print(f"\n   ├─ FTS Search:")
    try:
        fts_results = fts_search(query, k=k*2)
        print(f"   │  Results: {len(fts_results)} hits")
        for i, r in enumerate(fts_results[:2], 1):
            rank = r.get("fts_rank", "N/A")
            snippet = r["content"][:80].replace("\n", " ")
            print(f"   │  [{i}] (rank={rank}) {snippet}...")
    except Exception as e:
        print(f"   │  ❌ Error: {e}")
    
    # Vector
    print(f"\n   ├─ Vector Search:")
    try:
        vector_results = vector_search(query, k=k*2)
        print(f"   │  Results: {len(vector_results)} hits")
        for i, r in enumerate(vector_results[:2], 1):
            snippet = r["content"][:80].replace("\n", " ")
            print(f"   │  [{i}] {snippet}...")
    except Exception as e:
        print(f"   │  ❌ Error: {e}")
    
    # Hybrid
    print(f"\n   └─ Hybrid Search:")
    try:
        hybrid_results = hybrid_search(query, k=k*2)
        print(f"      Results: {len(hybrid_results)} hits")
        for i, r in enumerate(hybrid_results[:2], 1):
            snippet = r["content"][:80].replace("\n", " ")
            print(f"      [{i}] {snippet}...")
    except Exception as e:
        print(f"      ❌ Error: {e}")
    
    # 3. Routing decision
    print(f"\n3️⃣  ROUTING DECISION")
    print(f"   Mode '{mode}' will use:")
    if mode == "keyword":
        actual_results = fts_results if 'fts_results' in locals() else []
        print(f"   → fts_search() = {len(actual_results)} results")
    elif mode == "hybrid":
        actual_results = hybrid_results if 'hybrid_results' in locals() else []
        print(f"   → hybrid_search() = {len(actual_results)} results")
    else:
        actual_results = vector_results if 'vector_results' in locals() else []
        print(f"   → vector_search() = {len(actual_results)} results")
    
    # 4. Re-ranking check
    if actual_results:
        print(f"\n4️⃣  RE-RANKING")
        pairs = [(query, r["content"]) for r in actual_results[:3]]
        scores = _reranker.predict(pairs)
        for i, (score, r) in enumerate(zip(scores, actual_results[:3]), 1):
            print(f"   [{i}] Score={score:.3f} | {r['content'][:60]}...")


def test_database_state():
    """Check if documents are in the database."""
    print(f"\n{'='*70}")
    print(f"DATABASE STATE")
    print(f"{'='*70}")
    print(f"Collection: {COLLECTION_NAME}")
    
    try:
        store = get_vector_store()
        # Try a dummy search to verify connectivity
        results = store.similarity_search("test", k=1)
        print(f"✓ Vector store connected")
        print(f"  Sample search returned {len(results)} result(s)")
        
        if not results:
            print(f"  ⚠️  WARNING: Database appears empty!")
            print(f"  → Run: uv run app/ingestion/ingestion.py")
    except Exception as e:
        print(f"❌ Vector store error: {e}")


if __name__ == "__main__":
    test_database_state()
    
    test_queries = [
        "what is the DTI threshold for personal loans?",
        "DTI",
        "NPA classification borrower defaults",
        "leave policy employees",
    ]
    
    for query in test_queries:
        diagnose_query(query, k=5)
    
    print(f"\n{'='*70}\n")
