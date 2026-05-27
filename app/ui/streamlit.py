import streamlit as st
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from app.services.loan_agent import loan_agent

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Financial Assistant",
    layout="wide"
)

st.title("🏦 AI Financial Assistant")
st.markdown("Ask anything about loans, DTI, EMI, eligibility, policies, etc.")

# ---------------- SESSION STATE ----------------
if "chat" not in st.session_state:
    st.session_state.chat = []

# ---------------- CHAT HISTORY ----------------
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---------------- USER INPUT ----------------
query = st.chat_input("Ask your question...")

if query:

    # save user message
    st.session_state.chat.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.write(query)

    # ---------------- CALL AGENT ----------------
    result = loan_agent({
        "question": query
    })

    answer = result.get("answer", "")
    citations = result.get("citations", [])

    # save assistant message
    st.session_state.chat.append({
        "role": "assistant",
        "content": answer
    })

    # ---------------- ASSISTANT RESPONSE ----------------
    with st.chat_message("assistant"):
        st.write(answer)

        # ---------------- CITATIONS (FIXED - NO BIG FONT ISSUE) ----------------
        if citations:

            st.subheader("📚 Sources")

            for i, c in enumerate(citations, 1):

                title = c.get("title", "Source")
                url = c.get("url", "")
                content = c.get("content", "")

                # FORCE SAFE RENDERING (NO MARKDOWN HEADERS)
                st.markdown(f"**{i}. {title}**")
                st.markdown(f"🔗 {url}")

                # IMPORTANT: prevents markdown formatting like #, ##
                st.text(content)

                st.markdown("---")