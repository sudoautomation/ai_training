from fastapi import APIRouter
from app.schemas.loan_schema import LoanRequest
from app.services.loan_agent import loan_agent

router = APIRouter(
    prefix="/loan",
    tags=["Loan"]
)


@router.post("/ask")
def ask_loan(request: LoanRequest):
    return loan_agent(request.model_dump())