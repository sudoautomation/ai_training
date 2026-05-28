from pydantic import BaseModel
from typing import Optional


class BorrowerProfile(BaseModel):
    customer_id: Optional[str] = None
    age: Optional[int] = None
    monthly_income: Optional[float] = None
    existing_emis: Optional[float] = None
    requested_loan_amount: Optional[float] = None
    tenure_months: Optional[int] = None
    credit_score: Optional[int] = None
    past_defaults: Optional[int] = None


class LoanRequest(BaseModel):
    question: str
    borrower_profile: Optional[BorrowerProfile] = None