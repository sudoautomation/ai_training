# agents/tools.py
# Defines all agent tools using LangChain @tool decorator.
# search_keyword removed as a tool since Gemini generates well-formed
# queries that are better served by hybrid or semantic search.
# fts_search still exists in search.py and is used internally by hybrid_search.

from langchain_core.tools import tool

from app.services.dti_service import calculate_dti as _calculate_dti
from app.services.risk_service import evaluate_risk as _evaluate_risk
from app.retrieval.search import vector_search, hybrid_search


@tool
def search_semantic(query: str) -> str:
    """Semantic search for broad policy questions with no specific terms. Example: how does loan approval work."""
    results = vector_search(query)
    if not results:
        return "No relevant policy information found."
    return _format_context(results)


@tool
def search_hybrid(query: str) -> str:
    """Hybrid search when query contains specific loan terms like DTI, LTV, NPA, CIBIL, EMI, or product names. Example: what is DTI ratio, NPA classification rules."""
    results = hybrid_search(query)
    if not results:
        return "No relevant policy information found."
    return _format_context(results)


@tool
def assess_borrower(
    monthly_income: float,
    credit_score: int,
    existing_emis: float = 0,
    requested_loan_amount: float = 0,
    tenure_months: int = 0,
    past_defaults: int = 0,
) -> dict:
    """Calculate DTI and evaluate credit risk in one step. Use for borrower approval or risk assessment."""
    profile = {
        "monthly_income": monthly_income,
        "existing_emis": existing_emis,
        "requested_loan_amount": requested_loan_amount,
        "tenure_months": tenure_months,
        "credit_score": credit_score,
        "past_defaults": past_defaults,
    }

    dti_result = _calculate_dti(profile)

    if not dti_result:
        return {"error": "Could not calculate DTI. Check that monthly_income is non-zero."}

    risk_result = _evaluate_risk(profile, dti_result)

    return {
        "emi": dti_result.get("emi"),
        "dti": dti_result.get("dti"),
        "risk_score": risk_result.get("risk_score"),
        "risk_level": risk_result.get("risk_level"),
        "recommendation": risk_result.get("recommendation"),
    }


def _format_context(results: list[dict]) -> str:
    """
    Format retrieved chunks into context for the LLM.
    Content trimmed to 400 chars to reduce tokens sent back to Gemini.
    """
    context = []

    for i, doc in enumerate(results, 1):
        meta = doc.get("metadata", {})
        content = doc.get("content", "")[:400]
        context.append(
            f"Source {i} | File: {meta.get('source')} | Page: {meta.get('page')}\n{content}\n"
        )

    return "\n".join(context)


# 3 tools: 2 retrieval + 1 assessment
TOOLS = [search_semantic, search_hybrid, assess_borrower]
