import os
import warnings
from typing import Dict

import psycopg
from psycopg.rows import dict_row

from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

from app.retrieval.config import get_vector_store

# ---------------------------------------------------
# LOAD ENV
# ---------------------------------------------------

load_dotenv()

# ---------------------------------------------------
# DISABLE WARNINGS
# ---------------------------------------------------

warnings.filterwarnings("ignore")

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# ---------------------------------------------------
# DB CONNECTION
# ---------------------------------------------------

_raw_conn = os.getenv(
    "PG_CONNECTION_STRING",
    ""
).replace(
    "postgresql+psycopg",
    "postgresql"
)

# ---------------------------------------------------
# CROSS ENCODER RERANKER
# ---------------------------------------------------

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# ---------------------------------------------------
# VECTOR SEARCH
# ---------------------------------------------------


def vector_search(
    query: str,
    k: int = 15,
    filters: dict | None = None
):

    vector_store = get_vector_store()

    docs = vector_store.similarity_search(
        query=query,
        k=k,
        filter=filters
    )

    return [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "source": "vector"
        }
        for doc in docs
    ]


# ---------------------------------------------------
# POSTGRESQL FULL TEXT SEARCH
# ---------------------------------------------------


def fts_search(
    query: str,
    k: int = 15,
    collection_name: str = "credit_risk_assessment",
    filters: dict | None = None
):

    conditions = []

    if filters:

        for key, value in filters.items():

            if value is not None:

                conditions.append(
                    f"e.cmetadata->>'{key}' = '{value}'"
                )

    metadata_sql = ""

    if conditions:
        metadata_sql = " AND " + " AND ".join(conditions)

    sql = f"""
        SELECT
            e.document AS content,
            e.cmetadata AS metadata,
            ts_rank(
                to_tsvector('english', e.document),
                plainto_tsquery('english', %(query)s)
            ) AS rank
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c
            ON c.uuid = e.collection_id
        WHERE c.name = %(collection)s
        AND to_tsvector('english', e.document)
            @@ plainto_tsquery('english', %(query)s)
        {metadata_sql}
        ORDER BY rank DESC
        LIMIT %(k)s
    """

    with psycopg.connect(
        _raw_conn,
        row_factory=dict_row
    ) as conn:

        with conn.cursor() as cur:

            cur.execute(
                sql,
                {
                    "query": query,
                    "collection": collection_name,
                    "k": k
                }
            )

            rows = cur.fetchall()

    return [
        {
            "content": row["content"],
            "metadata": row["metadata"],
            "fts_rank": float(row["rank"]),
            "source": "fts"
        }
        for row in rows
    ]


# ---------------------------------------------------
# HYBRID RETRIEVAL
# ---------------------------------------------------


def hybrid_retrieval(
    query: str,
    borrower_context: Dict,
    k: int = 7
):

    # ------------------------------------------------
    # BORROWER CONTEXT INJECTION
    # ------------------------------------------------

    borrower_profile = f"""
    Borrower Profile:

    Credit Score: {borrower_context.get("credit_score")}
    DTI Ratio: {borrower_context.get("dti")}
    Annual Income: {borrower_context.get("income")}
    Existing Loans: {borrower_context.get("existing_loans")}
    Default History: {borrower_context.get("default_history")}
    Loan Product: {borrower_context.get("loan_product")}
    """

    enriched_query = borrower_profile + "\n\n" + query

    # ------------------------------------------------
    # METADATA FILTERS
    # ------------------------------------------------

    filters = {
        "loan_product": borrower_context.get("loan_product"),
        "policy_category": borrower_context.get("policy_category"),
        "content_type": borrower_context.get("content_type")
    }

    # ------------------------------------------------
    # VECTOR SEARCH
    # ------------------------------------------------

    vector_results = vector_search(
        query=enriched_query,
        k=15,
        filters=filters
    )

    # ------------------------------------------------
    # FTS SEARCH
    # ------------------------------------------------

    fts_results = fts_search(
        query=query,
        k=15,
        filters=filters
    )

    # ------------------------------------------------
    # HYBRID MERGING
    # ------------------------------------------------

    scores = {}
    chunk_map = {}

    # VECTOR SCORES (50%)

    for rank, item in enumerate(vector_results):

        key = item["content"][:150]

        score = 0.5 * (1 / (60 + rank + 1))

        scores[key] = scores.get(key, 0) + score

        chunk_map[key] = item

    # FTS SCORES (50%)

    for rank, item in enumerate(fts_results):

        key = item["content"][:150]

        score = 0.5 * (1 / (60 + rank + 1))

        scores[key] = scores.get(key, 0) + score

        chunk_map[key] = item

    # ------------------------------------------------
    # SORT MERGED RESULTS
    # ------------------------------------------------

    merged = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    retrieved_docs = [
        chunk_map[key]
        for key, _ in merged[:20]
    ]

    # ------------------------------------------------
    # CROSS ENCODER RERANKING
    # ------------------------------------------------

    pairs = [
        (enriched_query, doc["content"])
        for doc in retrieved_docs
    ]

    rerank_scores = reranker.predict(pairs)

    reranked = sorted(
        zip(retrieved_docs, rerank_scores),
        key=lambda x: x[1],
        reverse=True
    )

    # ------------------------------------------------
    # FINAL TOP-K
    # ------------------------------------------------

    final_docs = []

    for doc, score in reranked[:k]:

        doc["rerank_score"] = float(score)

        final_docs.append(doc)

    return final_docs


# ---------------------------------------------------
# TEST
# ---------------------------------------------------

if __name__ == "__main__":

    borrower = {
        "credit_score": 620,
        "dti": 48,
        "income": 55000,
        "existing_loans": 2,
        "default_history": True,
        "loan_product": "personal_loan",
        "policy_category": "eligibility",
        "content_type": "policy"
    }

    query = """
    Is this borrower eligible for loan approval?
    """

    results = hybrid_retrieval(
        query=query,
        borrower_context=borrower,
        k=7
    )

    print("\nTOP RETRIEVED RESULTS")
    print("=" * 80)

    for i, item in enumerate(results, 1):

        print(f"\nRESULT #{i}")
        print("-" * 80)

        print("SOURCE:", item.get("source"))

        print(
            "RERANK SCORE:",
            round(item.get("rerank_score", 0), 4)
        )

        print("METADATA:")
        print(item["metadata"])

        print("\nCONTENT:")
        print(item["content"][:700])