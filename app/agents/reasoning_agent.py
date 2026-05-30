# agents/reasoning_agent.py
# Drives the agentic loop using LangChain create_agent.
# Raises AgentError on failure so loan_routes.py can catch by type.

import json
from dotenv import load_dotenv
from langchain.agents import create_agent

from app.agents.tools import TOOLS
from app.exceptions import AgentError

load_dotenv()

SYSTEM_PROMPT = """
You are a loan policy assistant for a financial institution.

If asked who you are, respond only with:
I am a loan policy assistant. I can help you with loan-related queries.
Do not call any tool for identity questions.

For all other questions follow this pattern:
Thought: what does the question need
Action: call the right tool once
Observation: use the result to answer, do not call the same tool twice
Action: call a different tool only if genuinely needed, then answer

Tools:
- search_semantic: conversational policy questions
- search_hybrid: specific terms mixed with natural language like DTI limits NPA rules LTV thresholds
- assess_borrower: borrower approval eligibility or risk, always use this for profile assessment

Rules:
- Never fabricate policy numbers not in tool results
- Never pass borrower profile to search tools
- Use assess_borrower for any profile in the question
- Call each tool only once per question
- If search tool was used end answer with:
  Sources:
  - <filename> Page <number>
- Omit Sources if no search tool was used
"""

_agent = create_agent(
    model="google_genai:gemini-2.5-flash",
    tools=TOOLS,
    system_prompt=SYSTEM_PROMPT,
)


def generate_answer(question: str, profile: dict | None) -> tuple[str, dict, dict]:
    """
    Run the agent for a given question and optional borrower profile.

    Args:
        question: The user loan-related question.
        profile: Optional borrower profile dict from the request payload.

    Returns:
        A tuple of (answer, dti, risk).
    """
    input_message = _build_input(question, profile)

    try:
        response = _agent.invoke({
            "messages": [
                {"role": "user", "content": input_message}
            ]
        })
    except Exception as e:
        raise AgentError(f"Agent execution failed: {str(e)}") from e

    answer = response["messages"][-1].text
    dti, risk = _extract_tool_outputs(response["messages"])

    return answer, dti, risk


_PROFILE_FIELDS = {
    "monthly_income",
    "existing_emis",
    "requested_loan_amount",
    "tenure_months",
    "credit_score",
    "past_defaults",
}


def _build_input(question: str, profile: dict | None) -> str:
    """
    Build the input string for the agent.
    Strips unused profile fields to reduce tokens sent to Gemini.
    """
    if profile:
        clean_profile = {k: v for k, v in profile.items() if k in _PROFILE_FIELDS}
        return (
            f"Question: {question}\n\n"
            f"Note: Use profile only for assess_borrower. Do not pass to search tools.\n\n"
            f"Borrower Profile:\n{json.dumps(clean_profile, indent=2)}"
        )

    return f"Question: {question}"


def _extract_tool_outputs(messages: list) -> tuple[dict, dict]:
    """
    Extract dti and risk from assess_borrower tool result messages.
    """
    collected_dti = {}
    collected_risk = {}

    for message in messages:
        name = getattr(message, "name", None)
        content = getattr(message, "content", None)

        if not name or not content:
            continue

        if isinstance(content, str):
            try:
                content = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                continue

        if name == "assess_borrower":
            if isinstance(content, dict) and "error" not in content:
                collected_dti = {
                    "emi": content.get("emi"),
                    "dti": content.get("dti"),
                }
                collected_risk = {
                    "risk_score": content.get("risk_score"),
                    "risk_level": content.get("risk_level"),
                    "recommendation": content.get("recommendation"),
                }

    return collected_dti, collected_risk
