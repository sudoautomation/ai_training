# api/loan_routes.py
# Handles the /loan/ask endpoint.
# Catches custom exceptions by type for clean error classification.
# No string matching needed since exceptions are raised at the source.

from fastapi import APIRouter
from app.schemas.loan_schema import LoanRequest
from app.agents.loan_agent import loan_agent
from app.exceptions import DBConnectionError, AgentError

router = APIRouter(
    prefix="/loan",
    tags=["Loan"]
)

DB_ERROR_MESSAGE = "Our knowledge base is currently unavailable. Please try again shortly."
AGENT_ERROR_MESSAGE = "The AI assistant is currently unavailable. Please try again shortly."
GENERIC_ERROR_MESSAGE = "Something went wrong. Please try again."


@router.post("/ask")
def ask_loan(request: LoanRequest):
    """
    Accept a loan question with optional borrower profile.
    Returns agent answer, risk assessment, and dti.
    On failure returns a JSON error field the UI displays as a toast.
    """
    try:
        return loan_agent(request.model_dump())

    except DBConnectionError:
        return {"error": DB_ERROR_MESSAGE}

    except AgentError:
        return {"error": AGENT_ERROR_MESSAGE}

    except Exception:
        return {"error": GENERIC_ERROR_MESSAGE}
