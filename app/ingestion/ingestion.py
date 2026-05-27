from dotenv import load_dotenv

from app.ingestion.metadata import enrich_metadata
from app.ingestion.chunking import split_documents
from app.ingestion.storage import store_vector
from app.ingestion.loader import load_document

load_dotenv()


def ingest_pdf(file_path: str):

    print("Ingestion Started")

    # 1. Load document
    docs = load_document(file_path)
    print(f"Total documents/pages loaded: {len(docs)}")

    # 2. Metadata
    docs = enrich_metadata(docs, file_path)
    print(f"Metadata enriched: {len(docs)} documents")

    # 3. Chunking
    chunks = split_documents(docs)
    print(f"Total chunks: {len(chunks)}")
    
    if not chunks:
        print("ERROR: No chunks to store!")
        return
 
    # 4. Store embeddings in vector db
    print(f"Storing {len(chunks)} chunks to 'hr_support_desk' collection...")
    try:
        store_vector(chunks, collection_name="hr_support_desk")
        print("✓ All chunks stored successfully")
    except Exception as e:
        print(f"ERROR during storage: {e}")
        raise

    print("Ingestion Completed")


if __name__ == "__main__":

    ingest_pdf(
        "data/Capstone_Project_3_Intelligent_Credit_Risk_Assessment_FAQ.pdf"
        # "data\HR_Support_Desk_KnowledgeBase.pdf"
    )
# "data\HR_Support_Desk_KnowledgeBase.pdf"
