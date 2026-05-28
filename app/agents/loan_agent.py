# agents/loan_agent.py
# Entry point for a loan query. Delegates everything to the reasoning agent.
# No intent detection. No manual retrieval routing. The model decides what to call.
# This file only assembles the final response envelope.

from app.agents.citation_agent import build_citations
from app.agents.reasoning_agent import generate_answer


def loan_agent(payload: dict) -> dict:
    """
    Handle a loan query end to end.

    The reasoning agent runs the agentic loop and returns the answer
    along with whatever tool outputs were collected during the run.

    Args:
        payload: Dict with keys question (str) and optionally borrower_profile (dict).

    Returns:
        Dict with answer, risk, and citations.
    """

    question = payload.get("question", "")
    profile = payload.get("borrower_profile")

    answer, docs, dti, risk = generate_answer(question, profile)

    return {
        "answer": answer,
        "dti": dti,
        "risk": risk,
        "citations": build_citations(docs),
    }
