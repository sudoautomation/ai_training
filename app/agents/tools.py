# agents/tools.py
# Defines all agent tools using LangChain @tool decorator.
# Each tool wraps one service. No business logic lives here.
# tool_definitions.py and tool_executor.py are no longer needed.

from langchain_core.tools import tool

from app.services.dti_service import calculate_dti as _calculate_dti
from app.services.risk_service import evaluate_risk as _evaluate_risk
from app.services.retrieval_service import retrieve_context


@tool
def search_policy(query: str) -> str:
    """
    Search the loan policy knowledge base for RBI guidelines, eligibility
    criteria, DTI thresholds, LTV limits, NPA rules, and any other credit
    policy information. Use this when the question involves policy, guidelines,
    how something works, or why a rule exists.
    Pass only a plain search query string. Never pass borrower profile data here.
    """
    docs, context = retrieve_context(query)

    if not context.strip():
        return "No relevant policy information found for this query."

    return context


@tool
def calculate_dti(
    monthly_income: float,
    existing_emis: float = 0,
    requested_loan_amount: float = 0,
    tenure_months: int = 0,
) -> dict:
    """
    Calculate the Debt-to-Income ratio and estimated EMI for a borrower.
    Use this when you need to compute financial figures from the borrower profile.

    Args:
        monthly_income: Borrower monthly income in rupees.
        existing_emis: Total existing EMI obligations per month.
        requested_loan_amount: The loan amount being requested.
        tenure_months: Loan repayment tenure in months.
    """
    profile = {
        "monthly_income": monthly_income,
        "existing_emis": existing_emis,
        "requested_loan_amount": requested_loan_amount,
        "tenure_months": tenure_months,
    }

    result = _calculate_dti(profile)

    if not result:
        return {"error": "Could not calculate DTI. Check that monthly_income is non-zero."}

    return result


@tool
def evaluate_risk(
    credit_score: int,
    dti: float,
    past_defaults: int = 0,
) -> dict:
    """
    Evaluate the credit risk of a borrower and return a risk score,
    risk level, and approval recommendation. Use this when the question
    involves borrower assessment, approval eligibility, or credit risk.
    Always call calculate_dti first to get the dti value before calling this.

    Args:
        credit_score: Borrower CIBIL or credit score.
        dti: Debt-to-income ratio as a percentage, from calculate_dti.
        past_defaults: Number of past loan defaults. 0 means no defaults.
    """
    profile = {
        "credit_score": credit_score,
        "past_defaults": past_defaults,
    }

    dti_dict = {"dti": dti}

    return _evaluate_risk(profile, dti_dict)


# All tools collected for agent registration
TOOLS = [search_policy, calculate_dti, evaluate_risk]
