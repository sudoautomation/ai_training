"""Comprehensive test to verify ingestion → storage → retrieval flow."""

from app.retrieval.debug import count_documents_in_collection, list_all_collections
from app.retrieval.config import COLLECTION_NAME
from app.retrieval.search import detect_mode, fts_search, vector_search, hybrid_search


def test_flow():
    print("=" * 70)
    print("VECTOR STORE TEST")
    print("=" * 70)
    
    # 1. Check database state
    print(f"\n1️⃣  DATABASE STATE")
    print(f"   Target collection: {COLLECTION_NAME}")
    
    total = count_documents_in_collection()
    if total is None:
        print("   ❌ Cannot connect to database")
        return
    
    print(f"   Documents in target collection: {total}")
    
    print(f"\n   All collections in database:")
    collections = list_all_collections()
    if collections:
        for c in collections:
            marker = "✓" if c["name"] == COLLECTION_NAME else "○"
            print(f"      {marker} {c['name']}: {c['doc_count']} docs")
    
    if total == 0:
        print("\n   ⚠️  NO DOCUMENTS FOUND IN TARGET COLLECTION")
        print("   → Run ingest first: python app/ingestion/ingestion.py")
        return
    
    # 2. Test retrieval modes
    print(f"\n2️⃣  RETRIEVAL TEST")
    
    test_queries = [
        ("what is leave policy?", "Should be hybrid"),
        ("DTI", "Should be keyword"),
        ("full conversational query about employee benefits and policies", "Should be vector"),
    ]
    
    for query, expected_mode in test_queries:
        mode = detect_mode(query)
        print(f"\n   Query: '{query}'")
        print(f"   Mode:  {mode} ({expected_mode})")
        
        try:
            if mode == "keyword":
                results = fts_search(query, k=3)
            elif mode == "hybrid":
                results = hybrid_search(query, k=3)
            else:
                results = vector_search(query, k=3)
            
            print(f"   Results: {len(results)} hits")
            if results:
                for i, r in enumerate(results[:1], 1):
                    meta = r.get("metadata", {})
                    snippet = r["content"][:100]
                    print(f"      [{i}] {meta.get('source', 'N/A')} | {snippet}...")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    test_flow()
