import streamlit as st
from risk import render_risk


def render_chat():
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

            if msg.get("risk"):
                render_risk(msg["risk"])

            if msg.get("citations"):
                with st.expander("Sources"):
                    for c in msg["citations"]:
                        st.write(c["title"])