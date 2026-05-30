# ui/app.py
# Main Streamlit entry point.
# On successful responses appends the answer to chat.
# On error responses shows a toast notification instead of crashing.

import streamlit as st
from api_client import ask_loan
from chat import render_chat
from payload import build_payload
from sidebar import render_sidebar

st.set_page_config(page_title="AI Financial Assistant")

if "chat" not in st.session_state:
    st.session_state.chat = []

st.title("AI Financial Assistant")

render_sidebar()
render_chat()

query = st.chat_input("Ask...")

if query:
    st.session_state.chat.append({
        "role": "user",
        "content": query
    })

    payload = build_payload(query)

    with st.spinner("Thinking..."):
        result = ask_loan(payload)

    # If the result contains an error show it as a toast and do not append
    # a broken assistant message to the chat history
    if "error" in result:
        st.toast(result["error"], icon="✅")

    else:
        st.session_state.chat.append({
            "role": "assistant",
            "content": result["answer"],
            "risk": result.get("risk"),
        })

    st.rerun()
