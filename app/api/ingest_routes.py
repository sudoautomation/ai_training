# api/ingest_routes.py
# Exposes a single POST endpoint to upload and ingest a PDF file.
# File is saved temporarily, ingested into the vector store, then cleaned up.
# DB and general errors are mapped to friendly messages.

import os
import shutil
import tempfile

from fastapi import APIRouter, HTTPException, UploadFile, File

from app.ingestion.ingestion import ingest_pdf

router = APIRouter(
    prefix="/ingest",
    tags=["Ingestion"]
)

# Substrings used to detect DB connection failures
DB_ERROR_SIGNALS = [
    "connection refused",
    "could not connect",
    "psycopg",
    "pgvector",
    "operational error",
    "database",
]


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Accept a PDF upload and ingest it into the vector store.
    Access via /docs in FastAPI Swagger UI to upload manually.
    """

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, file.filename)

    try:
        with open(tmp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        ingest_pdf(tmp_path)

    except Exception as e:
        error_message = _classify_ingest_error(str(e).lower())
        raise HTTPException(status_code=500, detail=error_message)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return {
        "status": "success",
        "message": f"{file.filename} ingested successfully."
    }


def _classify_ingest_error(error_text: str) -> str:
    """
    Map a raw ingestion exception to a user-friendly message.
    """
    if any(signal in error_text for signal in DB_ERROR_SIGNALS):
        return "Could not connect to the database. Please ensure the vector store is running."

    return "Ingestion failed. Please check the file and try again."
