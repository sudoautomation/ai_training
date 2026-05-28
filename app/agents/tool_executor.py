# agents/tool_executor.py
# Responsible for executing a tool call by name and routing to the correct service.
# This is the only place that knows which service backs which tool.
# No LLM logic lives here. No retrieval logic lives here.

from app.services.dti_service import calculate_dti
from app.services.risk_service import evaluate_risk
from app.services.retrieval_service import retrieve_context


def execute_tool(tool_name: str, tool_args: dict) -> dict:
    """
    Execute a tool by name using the provided arguments.
    Returns a plain dict result that gets fed back to the agent.

    Args:
        tool_name: One of search_policy, calculate_dti, evaluate_risk.
        tool_args: Arguments parsed from the model's function call.

    Returns:
        A dict with the tool result or an error message.
    """

    if tool_name == "search_policy":
        return _run_search_policy(tool_args)

    if tool_name == "calculate_dti":
        return _run_calculate_dti(tool_args)

    if tool_name == "evaluate_risk":
        return _run_evaluate_risk(tool_args)

    return {"error": f"Unknown tool: {tool_name}"}


def _run_search_policy(args: dict) -> dict:
    """
    Calls the retrieval service with the model-generated query.
    Returns context text and raw docs for citation building.
    """
    query = args.get("query", "")

    if not query:
        return {"error": "search_policy requires a query argument"}

    docs, context = retrieve_context(query)

    return {
        "context": context,
        "docs": docs
    }


def _run_calculate_dti(args: dict) -> dict:
    """
    Builds a profile dict from the tool args and calls dti_service.
    The service expects the same keys so we pass args directly.
    """
    profile = {
        "monthly_income": args.get("monthly_income", 0),
        "existing_emis": args.get("existing_emis", 0),
        "requested_loan_amount": args.get("requested_loan_amount", 0),
        "tenure_months": args.get("tenure_months", 0),
    }

    result = calculate_dti(profile)

    if not result:
        return {"error": "Could not calculate DTI. Check that monthly_income is non-zero."}

    return result


def _run_evaluate_risk(args: dict) -> dict:
    """
    Builds the profile and dti dicts from tool args and calls risk_service.
    dti is passed as a nested dict to match evaluate_risk signature.
    """
    profile = {
        "credit_score": args.get("credit_score", 0),
        "past_defaults": args.get("past_defaults", 0),
    }

    dti = {
        "dti": args.get("dti", 100)
    }

    return evaluate_risk(profile, dti)
