# ui/chat.py
# Renders chat messages in the Streamlit UI.
# Each assistant message shows the answer text and optionally the risk panel.
# Sources are included by Gemini directly at the end of the answer text.

import streamlit as st
from risk import render_risk


def render_chat():
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

            if msg.get("risk"):
                render_risk(msg["risk"])
