from app.core.vector_store import create_vector_store


def store_vector(
    chunks,
    collection_name: str = "hr_support_desk"
):

    vector_store = create_vector_store(
        collection_name=collection_name
    )
    
    for chunk in chunks:
        vector_store.add_documents([chunk])
