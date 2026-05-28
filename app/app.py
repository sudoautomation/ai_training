from fastapi import FastAPI
from app.api.loan_routes import router as loan_router
from app.api.ingest_routes import router as ingest_router

app = FastAPI(title="AI Financial Assistant")

app.include_router(loan_router)
app.include_router(ingest_router)

@app.get("/")
def health():
    return {"status": "running"}