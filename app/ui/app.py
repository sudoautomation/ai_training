import streamlit as st

from sidebar import (
    render_sidebar
)

from chat import (
    render_chat
)

from payload import (
    build_payload
)

from api_client import (
    ask_loan
)


st.set_page_config(
    page_title="AI Financial Assistant"
)

if "chat" not in st.session_state:
    st.session_state.chat = []


st.title(
    "🏦 AI Financial Assistant"
)

render_sidebar()
render_chat()


query = st.chat_input(
    "Ask..."
)

if query:

    st.session_state.chat.append({

        "role":
        "user",

        "content":
        query
    })

    payload = build_payload(
        query
    )

    with st.spinner(
        "Thinking..."
    ):

        result = ask_loan(
            payload
        )

    st.session_state.chat.append({

        "role":
        "assistant",

        "content":
        result["answer"],

        "risk":
        result.get("risk"),

        "citations":
        result.get(
            "citations",
            []
        )
    })

    st.rerun()