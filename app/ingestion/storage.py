# ingestion/storage.py
# Embeds and stores document chunks in PGVector.
# All chunks are sent in a single add_documents call so OpenAI
# batches the embedding API requests internally instead of
# making one HTTP call per chunk.

from app.core.vector_store import create_vector_store


def store_vector(chunks: list, collection_name: str = "hr_support_desk") -> None:
    """
    Embed and store all chunks in PGVector in a single batch call.

    Args:
        chunks: List of LangChain Document objects to embed and store.
        collection_name: PGVector collection to store into.
    """
    vector_store = create_vector_store(collection_name=collection_name)

    # Single call lets OpenAI batch embeddings internally
    # much faster than calling add_documents per chunk
    vector_store.add_documents(chunks)
