"""
ingestion.py
------------
Single responsibility: read a PDF, chunk it, and WRITE embeddings
into the PGVector store.

Does NOT know how to query. Does NOT build the PGVector client itself.
It delegates both concerns to their respective modules.

Usage:
    python ingestion.py
    PDF_PATH=path/to/file.pdf python ingestion.py
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from vector_store import get_vector_store

load_dotenv()


def ingest_pdf(file_path: str, batch_size: int = 50) -> None:
    """
    Load a PDF, enrich metadata, chunk it, embed and store all chunks
    in PGVector.

    Args:
        file_path:  Absolute or relative path to the PDF file.
        batch_size: Chunks per upsert call. Larger = fewer round-trips.
    """

    # ── Validate ───────────────────────────────────────────────────────────
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found at: '{file_path}'")

    print("=" * 60)
    print("  INGESTION STARTED")
    print(f"  File      : {file_path}")
    print(f"  Batch size: {batch_size}")
    print("=" * 60)

    # ── Step 1: Load PDF ───────────────────────────────────────────────────
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    print(f"[1/4] PDF loaded       → {len(docs)} pages")

    # ── Step 2: Metadata enrichment ────────────────────────────────────────
    last_updated = os.path.getmtime(file_path)
    for doc in docs:
        doc.metadata.update({
            "source":             file_path,
            "document_extension": "pdf",
            # 0-based page index
            "page":               doc.metadata.get("page"),
            "last_updated":       last_updated,
        })
    print(f"[2/4] Metadata enriched → all pages tagged")

    # ── Step 3: Chunking ───────────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=60,
    )
    chunks = splitter.split_documents(docs)
    print(f"[3/4] Chunking done    → {len(chunks)} chunks created")

    # ── Step 4: Embed & store ──────────────────────────────────────────────
    vector_store = get_vector_store()

    total = len(chunks)
    vector_store.add_documents(chunks)
   

    print()  # newline after progress line
    print(f"[4/4] Storage complete → {total} chunks embedded & saved")
    print("=" * 60)
    print(f"  Ingestion finished successfully!")
    print(f"  Collection : {os.getenv('COLLECTION_NAME')}")
    print(
        f"  DB         : {os.getenv('PG_CONNECTION_STRING', '').split('@')[-1]}")
    print("=" * 60)


if __name__ == "__main__":
    PDF_PATH = os.getenv(
        "PDF_PATH",
        "data/Capstone_Project_3_Intelligent_Credit_Risk_Assessment_FAQ.pdf"
    )
    ingest_pdf(PDF_PATH)

