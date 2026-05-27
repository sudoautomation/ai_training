import streamlit as st
import requests

API_URL = "http://localhost:8000/chat"

st.set_page_config(
    page_title="Credit Risk Assistant",
    page_icon="🏦"
)

st.title("🏦 AI Credit Risk Assistant")

# session history
if "messages" not in st.session_state:
    st.session_state.messages = []

# display messages
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# user input
prompt = st.chat_input(
    "Ask about loans, DTI, eligibility..."
)

if prompt:

    # show user msg
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    # API call
    response = requests.post(
        API_URL,
        json={
            "message": prompt,
            "history": st.session_state.messages[:-1]
        }
    )

    assistant_reply = response.json()["response"]

    # show assistant msg
    with st.chat_message("assistant"):
        st.markdown(assistant_reply)

    # store assistant msg
    st.session_state.messages.append({
        "role": "assistant",
        "content": assistant_reply
    })