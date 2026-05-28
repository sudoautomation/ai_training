import re
import psycopg
from psycopg.rows import dict_row
from sentence_transformers import CrossEncoder

from app.retrieval.config import (
    RAW_CONN,
    COLLECTION_NAME,
    DEFAULT_K,
    HYBRID_VECTOR_WEIGHT,
    HYBRID_BM25_WEIGHT,
    RERANKER_MODEL,
    get_vector_store,
)

# ── Cross-encoder re-ranker (loaded once at import time) ────────────────────
_reranker = CrossEncoder(RERANKER_MODEL)

# ── Keyword patterns: RBI circulars, acronyms, numeric IDs ──────────────────
_KEYWORD_PATTERNS = [
    r"[A-Z]{2,}-\d{4}-\w+",  # e.g. RBI-2023-XYZ
    r"\b[A-Z]{2,5}\b",       # e.g. NPA, DTI, LTV, EMI
    r"\d{6,}",               # long numeric IDs
]
_KEYWORD_RE = re.compile("|".join(_KEYWORD_PATTERNS))


# ── Mode detection ───────────────────────────────────────────────────────────
def detect_mode(query: str) -> str:
    """Choose retrieval mode based on query structure.

    - 'keyword' → isolated acronyms / RBI codes / numeric IDs only (≤ 3 words AND matches pattern)
    - 'hybrid'  → short natural-language queries (≤ 3 words)
    - 'vector'  → full conversational / semantic queries (> 3 words)

    Priority: query length > keyword pattern (avoid routing long queries to FTS).
    """
    stripped = query.strip()
    word_count = len(stripped.split())
    has_keyword = bool(_KEYWORD_RE.search(stripped))

    # Long queries use vector, regardless of acronyms
    if word_count > 3:
        return "vector"

    # Short queries: use keyword only if pure acronym/code, else hybrid
    if has_keyword:
        return "keyword"

    return "hybrid"


# ── Metadata filter helper ───────────────────────────────────────────────────
def _matches_filter(metadata: dict, filters: dict) -> bool:
    """Return True if chunk metadata satisfies ALL provided filters.

    Filter values can be a string or list — any overlap = match.
    Unrecognised filter keys are ignored gracefully.
    """
    for key, value in filters.items():
        chunk_val = metadata.get(key)
        if chunk_val is None:
            return False
        # both sides normalised to lists for overlap check
        chunk_list = chunk_val if isinstance(chunk_val, list) else [chunk_val]
        filter_list = value if isinstance(value, list) else [value]
        if not any(v in chunk_list for v in filter_list):
            return False
    return True


# ── Re-ranker ────────────────────────────────────────────────────────────────
def rerank(query: str, results: list[dict], k: int = DEFAULT_K) -> list[dict]:
    """Cross-encoder re-ranking: score each (query, chunk) pair together,

    then reorder by true relevance and return top-k.
    """
    if not results:
        return results

    pairs = [(query, item["content"]) for item in results]
    scores = _reranker.predict(pairs)  # float score per pair

    ranked = sorted(
        zip(scores, results),
        key=lambda x: x[0],
        reverse=True,
    )
    return [item for _, item in ranked[:k]]


# ── Full-Text Search (BM25-like) ─────────────────────────────────────────────
def fts_search(
    query: str,
    k: int = DEFAULT_K,
    filters: dict = None,
) -> list[dict]:
    """PostgreSQL full-text search against the credit_risk_faq collection.

    Best for exact keyword matches: NPA, LTV, CIBIL codes, RBI circulars.
    Optionally filters by metadata fields (loan_product, risk_factor, etc.).
    """
    sql = """
        SELECT
            e.document   AS content,
            e.cmetadata  AS metadata,
            ts_rank(
                to_tsvector('english', e.document),
                plainto_tsquery('english', %(query)s)
            ) AS fts_rank
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c ON c.uuid = e.collection_id
        WHERE c.name = %(collection)s
          AND to_tsvector('english', e.document)
              @@ plainto_tsquery('english', %(query)s)
        ORDER BY fts_rank DESC
        LIMIT %(k)s;
    """
    with psycopg.connect(RAW_CONN, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, {"query": query, "collection": COLLECTION_NAME, "k": k * 2})
            rows = cur.fetchall()

    results = [
        {
            "content": row["content"],
            "metadata": row["metadata"],
            "fts_rank": round(float(row["fts_rank"]), 4),
        }
        for row in rows
    ]

    if filters:
        results = [r for r in results if _matches_filter(r["metadata"], filters)]

    # Re-rank with cross-encoder before returning
    results = rerank(query, results, k=k)
    return results


# ── Vector Search ────────────────────────────────────────────────────────────
def vector_search(
    query: str,
    k: int = DEFAULT_K,
    filters: dict = None,
) -> list[dict]:
    """Semantic similarity search via PGVector embeddings (cosine similarity).

    Best for full conversational / policy reasoning queries.
    Optionally filters by metadata fields post-retrieval.
    """
    # fetch extra to allow for post-filter drop
    fetch_k = k * 2 if filters else k
    docs = get_vector_store().similarity_search(query, k=fetch_k)

    results = [
        {"content": doc.page_content, "metadata": doc.metadata}
        for doc in docs
    ]

    if filters:
        results = [r for r in results if _matches_filter(r["metadata"], filters)]

    # Re-rank with cross-encoder before returning
    results = rerank(query, results, k=k)
    return results


# ── Hybrid Search (50/50 weighted RRF) ──────────────────────────────────────
def hybrid_search(
    query: str,
    k: int = DEFAULT_K,
    filters: dict = None,
) -> list[dict]:
    """Weighted Reciprocal Rank Fusion: 50% vector + 50% BM25/FTS.

    Spec: combine 50% BM25 keyword + 50% vector similarity.
    Optionally filters by metadata fields before fusion.
    """
    fetch_k = k * 2  # fetch more to absorb filter drops

    vector_docs = get_vector_store().similarity_search(query, k=fetch_k)
    fts_docs = fts_search(query, k=fetch_k, filters=filters)

    vector_results = [
        {"content": doc.page_content, "metadata": doc.metadata}
        for doc in vector_docs
    ]

    if filters:
        vector_results = [
            r for r in vector_results if _matches_filter(r["metadata"], filters)
        ]

    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    # ── Vector side: weighted 50% ────────────────────────────────────────────
    for rank, item in enumerate(vector_results):
        key = item["content"][:120]
        rrf_scores[key] = rrf_scores.get(key, 0) + HYBRID_VECTOR_WEIGHT / (60 + rank + 1)
        chunk_map[key] = item

    # ── FTS side: weighted 50% ───────────────────────────────────────────────
    for rank, item in enumerate(fts_docs):
        key = item["content"][:120]
        rrf_scores[key] = rrf_scores.get(key, 0) + HYBRID_BM25_WEIGHT / (60 + rank + 1)
        chunk_map[key] = item

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    hybrid_results = [chunk_map[key] for key, _ in ranked[:k]]

    # Final re-rank with cross-encoder for best relevance
    hybrid_results = rerank(query, hybrid_results, k=k)
    return hybrid_results