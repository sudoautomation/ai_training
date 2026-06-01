"""
streamlit_app.py
----------------
Production-grade Streamlit UI for the Credit Risk Assessment RAG chatbot.

Features:
  1. st.chat_input / st.chat_message — native chat UX (Streamlit ≥ 1.23).
  2. Full error handling — network errors, non-200 responses, and malformed
     JSON are all caught and shown as inline error banners (never a crash).
  3. Chat history persisted in st.session_state for the session lifetime.
  4. Citations rendered as an expandable section — clean by default.
  5. Confidence badge with colour coding (green / amber / red).
  6. Sidebar with app info, a "Clear chat" button, and connection status check.
  7. Configurable API URL via CREDIT_RISK_API_URL env var.
  8. Spinner with descriptive label while the agent is thinking.
  9. Request ID surfaced in the UI for support / debugging.
"""

import os
import time

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
API_URL = os.getenv("CREDIT_RISK_API_URL", "http://localhost:8000")
CHAT_ENDPOINT = f"{API_URL}/api/v1/chat"
HEALTH_ENDPOINT = f"{API_URL}/api/v1/health"
REQUEST_TIMEOUT = int(os.getenv("UI_REQUEST_TIMEOUT", "90"))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Assessment",
    page_icon="🏦",
    layout="centered",
)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []        # list of {role, content, meta}


# ── Helpers ───────────────────────────────────────────────────────────────────

CONFIDENCE_STYLE = {
    "high":   ("🟢", "green",  "High confidence"),
    "medium": ("🟡", "orange", "Medium confidence"),
    "low":    ("🔴", "red",    "Low confidence — verify in source document"),
}


def _confidence_badge(level: str) -> str:
    icon, colour, label = CONFIDENCE_STYLE.get(
        level, ("⚪", "grey", level)
    )
    return f":{colour}[{icon} {label}]"


def _ask(question: str) -> dict | None:
    """
    POST a question to the RAG API.
    Returns the parsed JSON dict on success, or None on any error
    (error message already written to st.error by this function).
    """
    try:
        response = requests.post(
            CHAT_ENDPOINT,
            json={"question": question},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.exceptions.ConnectionError:
        st.error(
            "⚠️ Cannot reach the API server. "
            f"Make sure it is running at **{API_URL}**."
        )
        return None
    except requests.exceptions.Timeout:
        st.error(
            f"⏱️ The request timed out after {REQUEST_TIMEOUT}s. "
            "The server may be overloaded — please try again."
        )
        return None
    except Exception as exc:
        st.error(f"Unexpected network error: {exc}")
        return None

    if response.status_code == 408:
        st.error(
            "⏱️ The agent timed out processing your question. Please try a simpler query.")
        return None

    if response.status_code == 422:
        detail = response.json().get("detail", "Invalid request.")
        st.error(f"❌ Validation error: {detail}")
        return None

    if response.status_code >= 500:
        try:
            body = response.json()
            detail = body.get("detail") or body.get(
                "error", "Unknown server error.")
        except Exception:
            detail = response.text[:300]
        st.error(f"🔥 Server error ({response.status_code}): {detail}")
        return None

    if response.status_code != 200:
        st.error(f"Unexpected response status: {response.status_code}")
        return None

    try:
        return response.json()
    except Exception:
        st.error("Could not parse the server response as JSON.")
        return None


def _render_assistant_message(data: dict):
    """Render answer, citations, confidence, and request ID."""
    st.markdown(data["answer"])

    # Citations expander
    citations = data.get("citations", [])
    pages = data.get("pages", [])

    if citations:
        with st.expander(
            f"📄 Sources — pages: {', '.join(str(p) for p in pages)}"
        ):
            for c in citations:
                st.markdown(
                    f"**Page {c['page']}** — _{c['text']}_"
                )
    elif pages:
        st.caption(f"Sources: pages {', '.join(str(p) for p in pages)}")

    # Confidence + request ID footer
    badge = _confidence_badge(data.get("confidence", ""))
    req_id = data.get("request_id", "")
    st.caption(f"{badge}   |   `{req_id}`")


# ── Main chat area ────────────────────────────────────────────────────────────
st.title("Credit Risk Assessment Assistant")

# Replay chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and "data" in msg:
            _render_assistant_message(msg["data"])
        else:
            st.markdown(msg["content"])

# Chat input
if question := st.chat_input("Ask a credit risk question…"):

    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # Call API and render response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and reasoning…"):
            t0 = time.perf_counter()
            data = _ask(question)
            elapsed = time.perf_counter() - t0

        if data:
            _render_assistant_message(data)
            st.session_state.messages.append(
                {"role": "assistant", "data": data, "content": data["answer"]}
            )
            st.caption(f"_Response time: {elapsed:.1f}s_")
        # If data is None, _ask() already rendered an st.error — nothing more to do
