# app/exceptions.py
# Custom exceptions for the AI Financial Assistant.
# Raised at the source so loan_routes.py can catch by type
# instead of guessing from raw error message strings.


class DBConnectionError(Exception):
    """
    Raised when the application cannot connect to the PostgreSQL vector store.
    Triggered in search.py when psycopg or PGVector throws a connection error.
    """
    pass


class AgentError(Exception):
    """
    Raised when the Gemini agent fails to complete execution.
    Triggered in reasoning_agent.py when create_agent invocation fails.
    """
    pass
