# retrieval/search.py
# Implements three search functions: fts_search, vector_search, hybrid_search.
# Raises DBConnectionError on any database connectivity failure so
# loan_routes.py can catch it by type instead of string matching.

import psycopg
from psycopg.rows import dict_row
from sentence_transformers import CrossEncoder

from app.exceptions import DBConnectionError
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
    return [item for _, item in ranked[:int(k)]]


def fts_search(
    query: str,
    k: int = DEFAULT_K,
    filters: dict = None,
    apply_rerank: bool = True,
) -> list[dict]:
    """
    PostgreSQL full-text search using ts_rank for keyword relevance scoring.
    apply_rerank=False when called internally by hybrid_search to avoid
    double reranking since hybrid_search reranks the fused results itself.
    Raises DBConnectionError on connection failure.
    """
    k = int(k)

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

    try:
        with psycopg.connect(RAW_CONN, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, {"query": query, "collection": COLLECTION_NAME, "k": k * 2})
                rows = cur.fetchall()
    except Exception as e:
        raise DBConnectionError(f"FTS search failed: {str(e)}") from e

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
    Raises DBConnectionError on connection failure.
    """
    k = int(k)
    fetch_k = k * 2 if filters else k

    try:
        docs = get_vector_store().similarity_search(query, k=fetch_k)
    except Exception as e:
        raise DBConnectionError(f"Vector search failed: {str(e)}") from e

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
    fts_search called with apply_rerank=False to avoid double reranking.
    Reranker runs once on the fused results.
    Raises DBConnectionError if either search path fails.
    """
    k = int(k)
    fetch_k = k * 2

    try:
        vector_docs = get_vector_store().similarity_search(query, k=fetch_k)
    except Exception as e:
        raise DBConnectionError(f"Vector search failed: {str(e)}") from e

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

    return rerank(query, hybrid_results, k=k)
