from app.core.vector_store import create_vector_store


def store_vector(
    chunks,
    collection_name: str = "hr_support_desk",
    batch_size: int = 5
):
    """
    Store chunks in batches to prevent API resource exhaustion.
    
    Args:
        chunks: List of document chunks to store
        collection_name: Target collection name
        batch_size: Number of chunks to process per batch (default 5)
    """

    vector_store = create_vector_store(
        collection_name=collection_name
    )
    
    total_chunks = len(chunks)
    print(f"Processing {total_chunks} chunks in batches of {batch_size}...")
    
    for batch_idx in range(0, total_chunks, batch_size):
        batch_end = min(batch_idx + batch_size, total_chunks)
        batch = chunks[batch_idx:batch_end]
        
        print(f"  Batch {batch_idx // batch_size + 1}: storing chunks {batch_idx + 1}-{batch_end}...")
        
        try:
            vector_store.add_documents(batch)
            print(f"    ✓ Batch complete")
        except Exception as e:
            print(f"    ❌ Error in batch: {e}")
            raise
    
    print(f"All {total_chunks} chunks stored successfully")
