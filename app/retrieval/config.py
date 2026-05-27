import os
from dotenv import load_dotenv
from app.core.vector_store import create_vector_store

load_dotenv()

# ── Collection name ─────────────────────────────────────────────────────────
COLLECTION_NAME = "hr_support_desk"

# ── Default retrieval k (spec: top-7) ───────────────────────────────────────
DEFAULT_K = 7

# ── Hybrid weights (spec: 50% BM25 keyword + 50% vector) ───────────────────
HYBRID_VECTOR_WEIGHT = 0.5
HYBRID_BM25_WEIGHT   = 0.5

# ── Re-ranker model (sentence-transformers cross-encoder) ───────────────────
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Raw psycopg connection string (stripped of SQLAlchemy prefix) ───────────
RAW_CONN = os.getenv("PG_CONNECTION_STRING", "").replace(
    "postgresql+psycopg2", "postgresql"
)


def get_vector_store():
    """Return a PGVector store pointed at the credit_risk_faq collection."""
    return create_vector_store(COLLECTION_NAME)
