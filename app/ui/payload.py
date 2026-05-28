import streamlit as st


def build_payload(query):
    payload = {"question": query}

    profile = st.session_state.get("borrower_profile", None)
    if profile:
        payload["borrower_profile"] = profile

    return payload