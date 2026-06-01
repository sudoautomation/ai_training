"""
vector_store.py
---------------
Single responsibility: build and return the PGVector client.

Both ingestion.py (writes) and retrieval.py (reads) import from here.
Neither depends on the other — this is the shared neutral foundation.

Environment variables required (set in .env):
  PG_CONNECTION_STRING  — e.g. postgresql+psycopg://user:pass@host:port/db
  COLLECTION_NAME       — e.g. credit_risk_docs
  EMBEDDING_MODEL       — e.g. text-embedding-3-small
"""

import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

# ── Config (read once at module level) ────────────────────────────────────────
PG_CONNECTION_STRING = os.getenv("PG_CONNECTION_STRING", "")
COLLECTION_NAME      = os.getenv("COLLECTION_NAME", "credit_risk_docs")
EMBEDDING_MODEL      = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


RAW_PG_CONNECTION_STRING = PG_CONNECTION_STRING.replace(
    "postgresql+psycopg", "postgresql"
)


def get_embeddings() -> OpenAIEmbeddings:
    """
    Return a configured OpenAI embeddings instance.
    Kept separate so it can be reused or swapped independently.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Add it to your .env file."
        )
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def get_vector_store(collection_name: str = COLLECTION_NAME) -> PGVector:
    """
    Build and return a PGVector client connected to the given collection.

    Args:
        collection_name: Override the default collection from .env.
                         Useful in tests or multi-tenant setups.

    Returns:
        A ready-to-use PGVector instance.

    Raises:
        EnvironmentError: If required env vars are missing.
    """
    if not PG_CONNECTION_STRING:
        raise EnvironmentError(
            "PG_CONNECTION_STRING is not set. "
            "Expected format: postgresql+psycopg://user:pass@host:port/db"
        )

    return PGVector(
        embeddings=get_embeddings(),
        collection_name=collection_name,
        connection=PG_CONNECTION_STRING,
    )