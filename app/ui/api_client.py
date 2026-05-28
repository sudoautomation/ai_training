import requests

BASE_URL = "http://localhost:8000"


def ask_loan(payload):
    response = requests.post(f"{BASE_URL}/loan/ask", json=payload)
    response.raise_for_status()
    return response.json()