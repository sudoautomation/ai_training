from fastapi import FastAPI
from app.api.loan_routes import router as loan_router

app = FastAPI(title="AI Financial Assistant")

app.include_router(loan_router)


@app.get("/")
def health():
    return {"status": "running"}