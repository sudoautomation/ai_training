# core/embeddings.py
# Creates the embedding model used for ingestion and retrieval.
# Uses OpenAI text-embedding-3-small which outputs 1536 dimensions
# matching the existing PGVector store configuration.

import os
from langchain_openai import OpenAIEmbeddings



def get_openai_api_key() -> str:
    return os.getenv("OPENAI_API_KEY")


def create_embeddings():
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=get_openai_api_key(),
        dimensions=1536,
    )
