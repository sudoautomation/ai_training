# retrieval/search.py
# Implements three search functions: fts_search, vector_search, hybrid_search.
# fts_search no longer runs the reranker since it is called internally by
# hybrid_search which reranks the final fused results anyway.
# Reranker only runs once per search call, at the end of each public function.

import psycopg
from psycopg.rows import dict_row
from sentence_transformers import CrossEncoder

from app.retrieval.config import (
    RAW_CONN,
    COLLECTION_NAME,
    DEFAULT_K,
    HYBRID_VECTOR_WEIGHT,
    HYBRID_FTS_WEIGHT,
    RERANKER_MODEL,
    get_vector_store,
)

# Cross-encoder reranker loaded once at import time
_reranker = CrossEncoder(RERANKER_MODEL)


def _matches_filter(metadata: dict, filters: dict) -> bool:
    """
    Return True if chunk metadata satisfies ALL provided filters.
    Filter values can be a string or list. Any overlap is a match.
    Unrecognised filter keys are ignored gracefully.
    """
    for key, value in filters.items():
        chunk_val = metadata.get(key)
        if chunk_val is None:
            return False
        chunk_list = chunk_val if isinstance(chunk_val, list) else [chunk_val]
        filter_list = value if isinstance(value, list) else [value]
        if not any(v in chunk_list for v in filter_list):
            return False
    return True


def rerank(query: str, results: list[dict], k: int = DEFAULT_K) -> list[dict]:
    """
    Cross-encoder reranking: score each (query, chunk) pair together,
    reorder by true relevance, and return top-k.
    """
    if not results:
        return results

    pairs = [(query, item["content"]) for item in results]
    scores = _reranker.predict(pairs)

    ranked = sorted(
        zip(scores, results),
        key=lambda x: x[0],
        reverse=True,
    )
    return [item for _, item in ranked[:k]]


def fts_search(
    query: str,
    k: int = DEFAULT_K,
    filters: dict = None,
    apply_rerank: bool = True,
) -> list[dict]:
    """
    PostgreSQL full-text search using ts_rank for keyword relevance scoring.
    Best for exact keyword matches: NPA, LTV, CIBIL codes, RBI circulars.

    apply_rerank is True when called directly as a tool.
    apply_rerank is False when called internally by hybrid_search to avoid
    double reranking since hybrid_search reranks the fused results itself.
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

    # Only rerank when called directly as a standalone tool
    if apply_rerank:
        return rerank(query, results, k=k)

    return results[:k]


def vector_search(
    query: str,
    k: int = DEFAULT_K,
    filters: dict = None,
) -> list[dict]:
    """
    Semantic similarity search via PGVector embeddings using cosine similarity.
    Best for full conversational and policy reasoning queries.
    Reranker runs once at the end.
    """
    fetch_k = k * 2 if filters else k
    docs = get_vector_store().similarity_search(query, k=fetch_k)

    results = [
        {"content": doc.page_content, "metadata": doc.metadata}
        for doc in docs
    ]

    if filters:
        results = [r for r in results if _matches_filter(r["metadata"], filters)]

    return rerank(query, results, k=k)


def hybrid_search(
    query: str,
    k: int = DEFAULT_K,
    filters: dict = None,
) -> list[dict]:
    """
    Weighted Reciprocal Rank Fusion combining 50% vector and 50% FTS results.
    fts_search is called with apply_rerank=False to avoid double reranking.
    Reranker runs once at the end on the fused results.
    """
    fetch_k = k * 1.5

    vector_docs = get_vector_store().similarity_search(query, k=fetch_k)

    # apply_rerank=False avoids running the reranker inside fts_search
    # since we rerank the fused results below
    fts_docs = fts_search(query, k=fetch_k, filters=filters, apply_rerank=False)

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

    # Vector side weighted 50%
    for rank, item in enumerate(vector_results):
        key = item["content"][:120]
        rrf_scores[key] = rrf_scores.get(key, 0) + HYBRID_VECTOR_WEIGHT / (60 + rank + 1)
        chunk_map[key] = item

    # FTS side weighted 50%
    for rank, item in enumerate(fts_docs):
        key = item["content"][:120]
        rrf_scores[key] = rrf_scores.get(key, 0) + HYBRID_FTS_WEIGHT / (60 + rank + 1)
        chunk_map[key] = item

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    hybrid_results = [chunk_map[key] for key, _ in ranked[:k]]

    # Single rerank on fused results
    return rerank(query, hybrid_results, k=k)
