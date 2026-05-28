# api/ingest_routes.py
# Exposes a single POST endpoint to upload and ingest a PDF file.
# File is saved temporarily, ingested into the vector store, then cleaned up.
# No business logic here. Delegates entirely to ingestion.py.

import os
import shutil
import tempfile

from fastapi import APIRouter, HTTPException, UploadFile, File

from app.ingestion.ingestion import ingest_pdf

router = APIRouter(
    prefix="/ingest",
    tags=["Ingestion"]
)


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Accept a PDF upload and ingest it into the vector store.

    The file is written to a temp directory, passed to ingest_pdf,
    then deleted regardless of success or failure.

    Access this via /docs in FastAPI's Swagger UI to upload manually.
    """

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Write the uploaded file to a temp path so ingest_pdf can read it from disk
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, file.filename)

    try:
        with open(tmp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        ingest_pdf(tmp_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

    finally:
        # Always clean up the temp file even if ingestion fails
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return {
        "status": "success",
        "message": f"{file.filename} ingested successfully."
    }
