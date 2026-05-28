# api/loan_routes.py
# Handles the /loan/ask endpoint.
# Catches agent and DB errors and returns a clean JSON error response
# instead of exposing raw tracebacks to the client.

from fastapi import APIRouter
from app.schemas.loan_schema import LoanRequest
from app.agents.loan_agent import loan_agent

router = APIRouter(
    prefix="/loan",
    tags=["Loan"]
)

# Error messages shown to the frontend
DB_ERROR_MESSAGE = "Our knowledge base is currently unavailable. Please try again shortly."
AGENT_ERROR_MESSAGE = "The AI assistant is currently unavailable. Please try again shortly."
GENERIC_ERROR_MESSAGE = "Something went wrong. Please try again."

# Substrings used to detect DB connection failures from exception messages
DB_ERROR_SIGNALS = [
    "connection refused",
    "could not connect",
    "psycopg",
    "pgvector",
    "pg_connection",
    "operational error",
    "database",
]

# Substrings used to detect Gemini/LLM API failures
AGENT_ERROR_SIGNALS = [
    "gemini",
    "google",
    "generativeai",
    "api key",
    "quota",
    "rate limit",
    "model",
]


@router.post("/ask")
def ask_loan(request: LoanRequest):
    """
    Accept a loan question with optional borrower profile.
    Returns the agent answer, risk assessment, and citations.
    On failure returns a JSON error field the UI can display as a toast.
    """
    try:
        return loan_agent(request.model_dump())

    except Exception as e:
        error_message = _classify_error(str(e).lower())
        return {"error": error_message}


def _classify_error(error_text: str) -> str:
    """
    Map a raw exception message to a user-friendly error string.
    Checks for DB signals first, then agent signals, then falls back to generic.
    """
    if any(signal in error_text for signal in DB_ERROR_SIGNALS):
        return DB_ERROR_MESSAGE

    if any(signal in error_text for signal in AGENT_ERROR_SIGNALS):
        return AGENT_ERROR_MESSAGE

    return GENERIC_ERROR_MESSAGE
