"""
retrieval.py
------------
Single responsibility: QUERY the vector store and return ranked results.

Does NOT load PDFs. Does NOT build the PGVector client itself.
It imports get_vector_store from vector_store.py — the neutral shared module.

"""


import os

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

from app.vector_store import (get_vector_store, RAW_PG_CONNECTION_STRING)

load_dotenv()

# ── Config (read once at module level) ────────────────────────────────────────
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "credit_risk_docs")


def retrieve_context(query: str, k: int = 4) -> list[dict]:
    """
    Alias for _hybrid_search — used by tools.py so the agent
    always calls a single stable name regardless of internal routing.
    """
    return _hybrid_search(query, k=k)


# ── Search strategies ─────────────────────────────────────────────────────────

def fts_search(
    query: str,
    k: int = 5,
    collection_name: str = COLLECTION_NAME,
) -> list[dict]:
    """
    PostgreSQL full-text search (tsvector / plainto_tsquery).
    Best for exact keyword matches: product codes, acronyms, IDs.
    """
    sql = """
        SELECT
            e.document   AS content,
            e.cmetadata  AS metadata,
            ts_rank(
                to_tsvector('english', e.document),
                plainto_tsquery('english', %(query)s)
            ) AS fts_rank
        FROM langchain_pg_embedding  e
        JOIN langchain_pg_collection c ON c.uuid = e.collection_id
        WHERE c.name = %(collection)s
          AND to_tsvector('english', e.document)
              @@ plainto_tsquery('english', %(query)s)
        ORDER BY fts_rank DESC
        LIMIT %(k)s;
    """
    with psycopg.connect(RAW_PG_CONNECTION_STRING, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql, {"query": query, "collection": collection_name, "k": k})
            rows = cur.fetchall()

    return [
        {
            "content":  row["content"],
            "metadata": row["metadata"],
            "fts_rank": round(float(row["fts_rank"]), 4),
        }
        for row in rows
    ]


def _hybrid_search(query: str, k: int = 5) -> list[dict]:
    """
    Reciprocal Rank Fusion (RRF) of vector + FTS results.
    Best for short queries that could match either semantically or literally.
    """
    vector_store = get_vector_store()
    vector_docs = vector_store.similarity_search(query, k=k)
    fts_docs = fts_search(query, k=k)
    print("[RETRIEVAL] Vector search results:" + str(vector_docs))
    print("[RETRIEVAL] FTS search results:" + str(fts_docs))

    rrf_scores: dict[str, float] = {}
    chunk_map:  dict[str, dict] = {}

    for rank, doc in enumerate(vector_docs):
        key = doc.page_content[:120]
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (60 + rank + 1)
        chunk_map[key] = {"content": doc.page_content,
                          "metadata": doc.metadata}

    for rank, item in enumerate(fts_docs):
        key = item["content"][:120]
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (60 + rank + 1)
        chunk_map[key] = {"content": item["content"],
                          "metadata": item["metadata"]}

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk_map[key] for key, _ in ranked[:k]]


# ── Manual test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # query = "how to calculate DTI?"
    query = "The system mentioned a similar past case in its recommendation — where did that come from and should I trust it?"
    results = _hybrid_search(query, k=4)
    print(f"\nTOP {len(results)} results for: '{query}'\n{'=' * 60}")
    for i, item in enumerate(results, 1):
        meta = item.get("metadata", {})
        print(f"\n[{i}] Source: {meta.get('source')} | Page: {meta.get('page')}")
        print(item["content"][:400])

        # python app/retrieval.py
