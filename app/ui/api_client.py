# ui/api_client.py
# Handles all HTTP communication with the FastAPI backend.
# Returns a structured dict on both success and failure so the UI
# always has a predictable shape to work with. Never raises exceptions.

import requests

BASE_URL = "http://localhost:8000"


def ask_loan(payload: dict) -> dict:
    """
    POST a loan question to the backend.

    On success returns the full response dict with answer, risk, citations.
    On any failure returns a dict with an error key containing a display message.
    """
    try:
        response = requests.post(f"{BASE_URL}/loan/ask", json=payload, timeout=60)

        # Backend returned a structured error response
        if not response.ok:
            return {"error": _http_error_message(response.status_code)}

        data = response.json()

        # Backend caught an internal error and returned it as JSON
        if "error" in data:
            return {"error": data["error"]}

        return data

    except requests.exceptions.ConnectionError:
        return {"error": "Cannot reach the server. Please check if the backend is running."}

    except requests.exceptions.Timeout:
        return {"error": "The request timed out. Please try again."}

    except Exception:
        return {"error": "An unexpected error occurred. Please try again."}


def _http_error_message(status_code: int) -> str:
    """
    Map HTTP status codes to user-friendly messages.
    """
    if status_code == 503:
        return "The service is currently unavailable. Please try again shortly."

    if status_code == 500:
        return "The server encountered an error. Please try again."

    if status_code == 422:
        return "The request could not be processed. Please check your inputs."

    return f"Unexpected server response ({status_code}). Please try again."
