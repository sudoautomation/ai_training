# agents/loan_agent.py
# Entry point for a loan query. Delegates everything to the reasoning agent.
# Citations are handled by Gemini directly in the answer text.
# This file only assembles the final response envelope.

from app.agents.reasoning_agent import generate_answer


def loan_agent(payload: dict) -> dict:
    """
    Handle a loan query end to end.

    The reasoning agent runs the agentic loop and returns the answer
    along with whatever tool outputs were collected during the run.

    Args:
        payload: Dict with keys question (str) and optionally borrower_profile (dict).

    Returns:
        Dict with answer, dti, and risk.
    """

    question = payload.get("question", "")
    profile = payload.get("borrower_profile")

    answer, dti, risk = generate_answer(question, profile)

    return {
        "answer": answer,
        "dti": dti,
        "risk": risk,
    }
