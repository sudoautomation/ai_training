
from dotenv import load_dotenv
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.routes import router

load_dotenv()


app = FastAPI(title="Credit Risk Assessment RAG")

app.include_router(router, prefix="/api/v1")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Something went wrong. Please try again."},
    )


@app.get("/health")
def health():
    return {"status": "ok"}
