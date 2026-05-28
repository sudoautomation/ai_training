from app.agents.citation_agent import build_citations
from app.agents.intent_agent import detect_intent
from app.agents.reasoning_agent import generate_answer
from app.services.dti_service import calculate_dti
from app.services.risk_service import evaluate_risk
from app.services.retrieval_service import retrieve_context


def loan_agent(payload):
    question = payload.get("question", "")
    profile = payload.get("borrower_profile")

    intent = detect_intent(question)
    docs = []
    context = ""

    if intent in ["policy", "calculation"]:
        docs, context = retrieve_context(question)

    dti = {}
    risk = {}

    if profile and intent == "assessment":
        dti = calculate_dti(profile)
        risk = evaluate_risk(profile, dti)

    answer = generate_answer(question, intent, profile, context, dti, risk)

    return {
        "answer": answer,
        "risk": risk,
        "citations": build_citations(docs),
    }