# retrieval/config.py
# Configuration for the retrieval pipeline.
# k reduced from 7 to 5 to lower tokens sent back to Gemini per search call.

import os
from dotenv import load_dotenv
from app.core.vector_store import create_vector_store

load_dotenv()

COLLECTION_NAME = "hr_support_desk"

# Reduced from 7 to 5 to lower token usage
DEFAULT_K = 5

# Hybrid weights: 50% FTS + 50% vector
HYBRID_VECTOR_WEIGHT = 0.5
HYBRID_FTS_WEIGHT = 0.5

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

RAW_CONN = os.getenv("PG_CONNECTION_STRING", "").replace(
    "postgresql+psycopg2", "postgresql"
)


def get_vector_store():
    """Return a PGVector store pointed at the credit_risk_faq collection."""
    return create_vector_store(COLLECTION_NAME)
