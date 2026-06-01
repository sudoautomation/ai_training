from langchain.tools import tool
from app.retrieval import retrieve_context


@tool
def retrieve_credit_risk_context(query: str) -> dict:
    """Retrieve relevant credit risk context for the given query."""

    docs = retrieve_context(query)
    return {
        "evidence": [
            {
                "page": doc["metadata"].get("page"),
                "content": doc["content"]
            }
            for doc in docs
        ]
    }
