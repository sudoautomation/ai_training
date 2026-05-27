import re
import os
from dotenv import load_dotenv

import psycopg
from psycopg.rows import dict_row

from app.core.vector_store import create_vector_store

load_dotenv()

# -----------------------------
# DB CONNECTION
# -----------------------------
_row_conn = os.getenv("PG_CONNECTION_STRING", "").replace(
    "postgresql+psycopg", "postgresql"
)

# -----------------------------
# KEYWORD PATTERNS
# -----------------------------
_KEYWORD_PATTERNS = [
    r"[A-Z]{2,}-\d{4}-\w+",
    r"\b[A-Z]{2,5}\b",
    r"\d{6,}",
]

_KEYWORD_RE = re.compile("|".join(_KEYWORD_PATTERNS))


# -----------------------------
# MODE DETECTION
# -----------------------------
def _detect_mode(query: str) -> str:
    stripped = query.strip()

    if _KEYWORD_RE.search(stripped):
        return "keyword"

    if len(stripped.split()) <= 3:
        return "hybrid"

    return "vector"


# -----------------------------
# MAIN RETRIEVAL FUNCTION
# -----------------------------
def query_documents(query: str, k: int = 5) -> list[dict]:

    mode = _detect_mode(query)

    if mode == "keyword":
        return fts_search(query, k=k)

    if mode == "hybrid":
        return _hybrid_search(query, k=k)

    # vector search
    vector_store = create_vector_store()
    docs = vector_store.similarity_search(query, k=k)

    return [
        {
            "content": doc.page_content,
            "metadata": doc.metadata
        }
        for doc in docs
    ]


# -----------------------------
# FULL TEXT SEARCH (FTS)
# -----------------------------
def fts_search(
    query: str,
    k: int = 5,
    collection_name: str = "hr_support_desk"
) -> list[dict]:

    sql = """
       SELECT
           e.document  AS content,
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

    with psycopg.connect(_row_conn, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, {
                "query": query,
                "collection": collection_name,
                "k": k
            })
            rows = cur.fetchall()

    return [
        {
            "content": row["content"],
            "metadata": row["metadata"],
            "fts_rank": round(float(row["fts_rank"]), 4),
        }
        for row in rows
    ]


# -----------------------------
# HYBRID SEARCH
# -----------------------------
def _hybrid_search(query: str, k: int = 5) -> list[dict]:

    vector_store = create_vector_store()

    vector_docs = vector_store.similarity_search(query, k=k)
    fts_docs = fts_search(query, k=k)

    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    # Vector results
    for rank, doc in enumerate(vector_docs):
        key = doc.page_content[:120]

        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (60 + rank + 1)

        chunk_map[key] = {
            "content": doc.page_content,
            "metadata": doc.metadata
        }

    # FTS results
    for rank, item in enumerate(fts_docs):
        key = item["content"][:120]

        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (60 + rank + 1)

        chunk_map[key] = {
            "content": item["content"],
            "metadata": item["metadata"]
        }

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    return [chunk_map[key] for key, _ in ranked[:k]]


# -----------------------------
# TEST
# -----------------------------
if __name__ == "__main__":
    query = "calculation for dti ratio?"
    results = query_documents(query, k=5)

    print(f"\nTop {len(results)} results:\n{'=' * 60}")

    for i, item in enumerate(results, 1):
        metadata = item.get("metadata", {})

        print(f"\n[{i}] source: {metadata.get('source')} | page: {metadata.get('page')}")
        print(item["content"][:400])

'''import re
import os
from dotenv import load_dotenv

import psycopg
from psycopg.rows import dict_row

from app.core.db import get_vector_store

load_dotenv()


# -----------------------------
# DB CONNECTION
# -----------------------------
_row_conn = os.getenv(
    "PG_CONNECTION_STRING",
    ""
).replace(
    "postgresql+psycopg",
    "postgresql"
)


# -----------------------------
# COLLECTION NAME
# IMPORTANT:
# Change this to your
# loan collection name
# -----------------------------
COLLECTION_NAME = "hr_support_desk"


# -----------------------------
# KEYWORD PATTERNS
# -----------------------------
_KEYWORD_PATTERNS = [
    r"[A-Z]{2,}-\d{4}-\w+",
    r"\b[A-Z]{2,5}\b",
    r"\d{6,}",
]

_KEYWORD_RE = re.compile(
    "|".join(_KEYWORD_PATTERNS)
)


# -----------------------------
# MODE DETECTION
# -----------------------------
def _detect_mode(query: str) -> str:

    stripped = query.strip()

    # Keyword-heavy query
    if _KEYWORD_RE.search(stripped):
        return "keyword"

    # Short query → hybrid
    if len(stripped.split()) <= 3:
        return "hybrid"

    # Long query → vector
    return "vector"


# -----------------------------
# MAIN RETRIEVAL FUNCTION
# -----------------------------
def query_documents(
    query: str,
    k: int = 5
) -> list[dict]:

    mode = _detect_mode(query)

    print(f"\n🔍 Retrieval Mode: {mode}")

    # -------------------------
    # KEYWORD SEARCH
    # -------------------------
    if mode == "keyword":
        return fts_search(
            query,
            k=k
        )

    # -------------------------
    # HYBRID SEARCH
    # -------------------------
    if mode == "hybrid":
        return _hybrid_search(
            query,
            k=k
        )

    # -------------------------
    # VECTOR SEARCH
    # -------------------------
    vector_store = get_vector_store()

    docs = vector_store.similarity_search(
        query,
        k=k
    )

    results = []

    for doc in docs:

        metadata = doc.metadata or {}

        results.append({
            "content": doc.page_content,
            "metadata": {
                "source": metadata.get(
                    "source",
                    "loan_policy.pdf"
                ),
                "page": metadata.get(
                    "page",
                    "Unknown"
                )
            }
        })

    return results


# -----------------------------
# FULL TEXT SEARCH (FTS)
# -----------------------------
def fts_search(
    query: str,
    k: int = 5,
    collection_name: str = COLLECTION_NAME
) -> list[dict]:

    sql = """
       SELECT
           e.document AS content,
           e.cmetadata AS metadata,

           ts_rank(
               to_tsvector(
                   'english',
                   e.document
               ),
               plainto_tsquery(
                   'english',
                   %(query)s
               )
           ) AS fts_rank

       FROM langchain_pg_embedding e

       JOIN langchain_pg_collection c
       ON c.uuid = e.collection_id

       WHERE c.name = %(collection)s

       AND to_tsvector(
            'english',
            e.document
       )
       @@ plainto_tsquery(
            'english',
            %(query)s
       )

       ORDER BY fts_rank DESC
       LIMIT %(k)s;
    """

    with psycopg.connect(
        _row_conn,
        row_factory=dict_row
    ) as conn:

        with conn.cursor() as cur:

            cur.execute(sql, {
                "query": query,
                "collection": collection_name,
                "k": k
            })

            rows = cur.fetchall()

    results = []

    for row in rows:

        metadata = row.get(
            "metadata",
            {}
        ) or {}

        results.append({
            "content": row["content"],
            "metadata": {
                "source": metadata.get(
                    "source",
                    "loan_policy.pdf"
                ),
                "page": metadata.get(
                    "page",
                    "Unknown"
                )
            },
            "fts_rank": round(
                float(row["fts_rank"]),
                4
            )
        })

    return results


# -----------------------------
# HYBRID SEARCH
# -----------------------------
def _hybrid_search(
    query: str,
    k: int = 5
) -> list[dict]:

    vector_store = get_vector_store()

    vector_docs = (
        vector_store
        .similarity_search(
            query,
            k=k
        )
    )

    fts_docs = fts_search(
        query,
        k=k
    )

    rrf_scores = {}
    chunk_map = {}

    # -------------------------
    # VECTOR RESULTS
    # -------------------------
    for rank, doc in enumerate(vector_docs):

        key = doc.page_content[:120]

        rrf_scores[key] = (
            rrf_scores.get(key, 0)
            + 1 / (60 + rank + 1)
        )

        metadata = doc.metadata or {}

        chunk_map[key] = {
            "content": doc.page_content,
            "metadata": {
                "source": metadata.get(
                    "source",
                    "loan_policy.pdf"
                ),
                "page": metadata.get(
                    "page",
                    "Unknown"
                )
            }
        }

    # -------------------------
    # FTS RESULTS
    # -------------------------
    for rank, item in enumerate(fts_docs):

        key = item["content"][:120]

        rrf_scores[key] = (
            rrf_scores.get(key, 0)
            + 1 / (60 + rank + 1)
        )

        chunk_map[key] = item

    ranked = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        chunk_map[key]
        for key, _ in ranked[:k]
    ]


# -----------------------------
# TEST
# -----------------------------
if __name__ == "__main__":

    query = (
        "What credit score qualifies "
        "a borrower for low-risk classification?"
    )

    results = query_documents(
        query,
        k=5
    )

    print(
        f"\nTop {len(results)} results:\n"
        + "=" * 60
    )

    for i, item in enumerate(
        results,
        1
    ):

        metadata = item.get(
            "metadata",
            {}
        )

        print(
            f"\n[{i}] "
            f"source: "
            f"{metadata.get('source')} "
            f"| "
            f"page: "
            f"{metadata.get('page')}"
        )

        print(
            item["content"][:400]
        )'''