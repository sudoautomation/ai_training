from dotenv import load_dotenv

from app.ingestion.metadata import enrich_metadata
from app.ingestion.chunking import split_documents
from app.ingestion.storage import store_vector
from app.ingestion.loader import load_document
from app.core.vector_store import create_vector_store

load_dotenv()


def ingest_pdf(file_path: str):
    print("Ingestion Started")

    # 1. Load document
    docs = load_document(file_path)
    print(f"Total documents/pages loaded: {len(docs)}")

    # 2. Metadata
    docs = enrich_metadata(docs, file_path)

    # 3. Chunking
    chunks = split_documents(docs)
    print(f"Total chunks: {len(chunks)}")

    # 4. create vector
    create_vector_store("hr_support_desk", pre_delete_collection=True)

    # 5. store embeddings in vector db
    store_vector(chunks)

    print("Ingestion Completed")


if __name__ == "__main__":
    ingest_pdf("data/Capstone_Project_3_Intelligent_Credit_Risk_Assessment_FAQ.pdf")

# Alternative path or commands for reference:
# "data\HR_Support_Desk_KnowledgeBase.pdf"
# $env:PYTHONPATH="."; uv run app/ingestion/ingestion.py