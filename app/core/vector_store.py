import os

from dotenv import load_dotenv

from langchain_postgres import PGVector
from app.core.embeddings import create_embeddings

load_dotenv()

def get_pg_connection_string() -> str:
    return os.getenv("PG_CONNECTION_STRING")

# =========================
# Vector Store Factory
# =========================

def create_vector_store(
    collection_name: str = "hr_support_desk"
):
    return PGVector(
        collection_name=collection_name,
        connection=get_pg_connection_string(),
        embeddings=create_embeddings(),
        use_jsonb=True
    )
    