"""Analyze retrieved document quality and relevance."""

from app.retrieval.search import (
    detect_mode, fts_search, vector_search, hybrid_search, 
    rerank, _reranker, COLLECTION_NAME
)

def analyze_query(query: str, k: int = 7):
    """Deep analysis of what gets retrieved and how it's ranked."""
    print(f"\n{'='*80}")
    print(f"QUERY: {query}")
    print(f"{'='*80}")
    
    mode = detect_mode(query)
    print(f"Mode: {mode}\n")
    
    # Get raw results before re-ranking
    if mode == "keyword":
        raw_results = fts_search(query, k=k*2)
        print(f"FTS raw results: {len(raw_results)} hits")
    elif mode == "hybrid":
        raw_results = hybrid_search(query, k=k*2)
        print(f"Hybrid raw results: {len(raw_results)} hits")
    else:
        raw_results = vector_search(query, k=k*2)
        print(f"Vector raw results: {len(raw_results)} hits")
    
    if not raw_results:
        print("❌ NO RESULTS FOUND")
        return
    
    # Score with re-ranker
    print(f"\n{'─'*80}")
    print("RE-RANKER ANALYSIS:")
    print(f"{'─'*80}")
    
    pairs = [(query, r["content"]) for r in raw_results]
    scores = _reranker.predict(pairs)
    
    # Sort by score
    scored_results = list(zip(scores, raw_results))
    scored_results.sort(key=lambda x: x[0], reverse=True)
    
    print(f"\nTop {min(k, len(scored_results))} by re-ranker score:\n")
    
    for i, (score, doc) in enumerate(scored_results[:k], 1):
        content = doc["content"][:120].replace("\n", " ")
        meta = doc.get("metadata", {})
        source = meta.get("source", "N/A")
        page = meta.get("page", "N/A")
        
        print(f"[{i}] SCORE: {score:.2f}")
        print(f"    Source: {source} | Page: {page}")
        print(f"    Content: {content}...")
        print()
    
    # Check score distribution
    print(f"{'─'*80}")
    print("SCORE DISTRIBUTION:")
    print(f"{'─'*80}")
    print(f"Max score:    {max(scores):.3f}")
    print(f"Min score:    {min(scores):.3f}")
    print(f"Mean score:   {sum(scores)/len(scores):.3f}")
    print(f"Median score: {sorted(scores)[len(scores)//2]:.3f}")
    
    # Check for negative scores (bad sign)
    negative = sum(1 for s in scores if s < 0)
    if negative > 0:
        print(f"⚠️  {negative}/{len(scores)} documents have NEGATIVE scores!")


if __name__ == "__main__":
    # Test with credit risk queries
    queries = [
        "what is the DTI threshold for personal loans?",
        "DTI",
        "How do I assess loan repayment capacity?",
        "what factors determine credit risk?",
    ]
    
    for q in queries:
        analyze_query(q, k=5)
    
    print("\n" + "="*80 + "\n")
