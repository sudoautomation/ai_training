import re
import os
from dotenv import load_dotenv
import psycopg
from psycopg.rows import dict_row
from app.retrieval.config import get_vector_store

load_dotenv()

_raw_conn = os.getenv("PG_CONNECTION_STRING", "").replace(
    "postgresql+psycopg", "postgresql"
)

_KEYWORD_PATTERNS = [
    r"[A-Z]{2,}-\d{4}-\w+",
    r"\b[A-Z]{2,5}\b",
    r"\d{6,}",
]

_KEYWORD_RE = re.compile("|".join(_KEYWORD_PATTERNS))


def _detect_mode(query: str) -> str:
    stripped = query.strip()

    if _KEYWORD_RE.search(stripped):
        return "keyword"

    if len(stripped.split()) <= 3:
        return "hybrid"

    return "vector"


def query_document(query: str, k: int = 5) -> list[dict]:
    mode = _detect_mode(query)

    if mode == "keyword":
        return fts_search(query, k=k)

    if mode == "hybrid":
        return _hybrid_search(query, k=k)

    vector_store = get_vector_store()
    docs = vector_store.similarity_search(query, k=k)

    return [
        {"content": doc.page_content, "metadata": doc.metadata}
        for doc in docs
    ]


def fts_search(query: str, k: int = 5, collection_name: str = "hr_support_desk"):
    sql = """
        SELECT
            e.document AS content,
            e.cmetadata AS metadata,
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

    with psycopg.connect(_raw_conn, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                {
                    "query": query,
                    "collection": collection_name,
                    "k": k,
                },
            )
            rows = cur.fetchall()

    return [
        {
            "content": row["content"],
            "metadata": row["metadata"],
            "fts_rank": round(float(row["fts_rank"]), 4),
        }
        for row in rows
    ]


def _hybrid_search(query: str, k: int = 5):
    vector_store = get_vector_store()

    vector_docs = vector_store.similarity_search(query, k=k)
    fts_docs = fts_search(query, k=k)

    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    for rank, doc in enumerate(vector_docs):
        key = doc.page_content[:120]
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (60 + rank + 1)
        chunk_map[key] = {
            "content": doc.page_content,
            "metadata": doc.metadata,
        }

    for rank, item in enumerate(fts_docs):
        key = item["content"][:120]
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (60 + rank + 1)
        chunk_map[key] = {
            "content": item["content"],
            "metadata": item["metadata"],
        }

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    return [chunk_map[key] for key, _ in ranked[:k]]


if __name__ == "__main__":
    query = "dti ratio?"
    results = query_document(query, k=5)

    print(f"\nTOP {len(results)} results for: '{query}'\n{'='*60}")

    for i, item in enumerate(results, 1):
        metadata = item.get("metadata", {})
        print(f"\n[{i}] Source: {metadata.get('source')} | Page: {metadata.get('page')}")
        print(item["content"][:400])