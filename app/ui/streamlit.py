import streamlit as st
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from app.services.loan_agent import loan_agent

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Financial Assistant",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 AI Financial Assistant")
st.markdown("Ask anything about loans, DTI, EMI, eligibility, or submit a borrower profile for a full risk assessment.")

# ---------------- SESSION STATE ----------------
if "chat" not in st.session_state:
    st.session_state.chat = []

if "borrower_profile" not in st.session_state:
    st.session_state.borrower_profile = None

if "show_profile_form" not in st.session_state:
    st.session_state.show_profile_form = False


# =====================================================
# SIDEBAR — BORROWER PROFILE FORM
# =====================================================
with st.sidebar:

    st.header("📋 Borrower Profile")
    st.caption("Fill this for a full risk assessment. Leave blank for general Q&A.")

    with st.expander("Enter Borrower Details", expanded=st.session_state.show_profile_form):

        customer_id =              st.text_input("Customer ID", placeholder="e.g. BORR001")
        age =                      st.number_input("Age", min_value=18, max_value=75, value=30)
        employment_type =          st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Business", "Freelance", "Other"])
        monthly_income =           st.number_input("Monthly Income (₹)", min_value=0, value=0, step=1000)
        existing_emis =            st.number_input("Existing EMIs / Month (₹)", min_value=0, value=0, step=500)
        requested_loan_amount =    st.number_input("Requested Loan Amount (₹)", min_value=0, value=0, step=10000)
        loan_type =                st.selectbox("Loan Type", ["Personal Loan", "Home Loan", "Auto Loan", "Business Loan", "Education Loan", "Gold Loan"])
        tenure_months =            st.number_input("Tenure (months)", min_value=1, max_value=360, value=60)
        credit_score =             st.number_input("Credit Score (CIBIL)", min_value=300, max_value=900, value=700)
        past_defaults =            st.number_input("Past Defaults", min_value=0, max_value=20, value=0)
        employment_stability_years = st.number_input("Employment Stability (years)", min_value=0, max_value=40, value=2)
        bank_balance_avg =         st.number_input("Avg Bank Balance (₹)", min_value=0, value=0, step=1000)

        col1, col2 = st.columns(2)

        with col1:
            if st.button("✅ Set Profile", use_container_width=True):
                st.session_state.borrower_profile = {
                    "customer_id":                customer_id,
                    "age":                        age,
                    "employment_type":            employment_type,
                    "monthly_income":             monthly_income,
                    "existing_emis":              existing_emis,
                    "requested_loan_amount":      requested_loan_amount,
                    "loan_type":                  loan_type,
                    "tenure_months":              tenure_months,
                    "credit_score":               credit_score,
                    "past_defaults":              past_defaults,
                    "employment_stability_years": employment_stability_years,
                    "bank_balance_avg":           bank_balance_avg
                }
                st.rerun()

        with col2:
            if st.button("🗑️ Clear Profile", use_container_width=True):
                st.session_state.borrower_profile = None
                st.rerun()

    # Show active profile summary
    if st.session_state.borrower_profile:
        p = st.session_state.borrower_profile
        st.success(f"**Active Profile:** {p.get('customer_id', 'N/A')}")
        st.caption(
            f"Income: ₹{p.get('monthly_income', 0):,} | "
            f"Loan: ₹{p.get('requested_loan_amount', 0):,} | "
            f"CIBIL: {p.get('credit_score', 0)}"
        )
    else:
        st.info("No borrower profile loaded — Q&A mode active.")

    st.divider()

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat = []
        st.rerun()


# =====================================================
# RISK SCORECARD RENDERER
# =====================================================
def render_risk_scorecard(risk: dict):

    score =          risk.get("risk_score", "N/A")
    level =          risk.get("risk_level", "N/A")
    recommendation = risk.get("recommendation", "N/A")
    confidence =     risk.get("confidence", "N/A")

    color_map = {
        "LOW":    "green",
        "MEDIUM": "orange",
        "HIGH":   "red"
    }

    rec_color_map = {
        "APPROVE":              "green",
        "CONDITIONAL APPROVE":  "orange",
        "REJECT":               "red"
    }

    level_color = color_map.get(level, "gray")
    rec_color =   rec_color_map.get(recommendation, "gray")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Risk Score", f"{score} / 10")

    with col2:
        st.markdown(
            f"**Risk Level**<br>"
            f"<span style='color:{level_color}; font-size:1.2rem; font-weight:bold'>{level}</span>",
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"**Recommendation**<br>"
            f"<span style='color:{rec_color}; font-size:1.1rem; font-weight:bold'>{recommendation}</span>",
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"**Confidence**<br>"
            f"<span style='font-size:1.1rem; font-weight:bold'>{confidence}</span>",
            unsafe_allow_html=True
        )


# =====================================================
# CHAT HISTORY RENDERER
# =====================================================
for msg in st.session_state.chat:

    with st.chat_message(msg["role"]):

        st.write(msg["content"])

        # Re-render risk scorecard from history if present
        if msg.get("risk"):
            render_risk_scorecard(msg["risk"])

        # Re-render citations from history if present
        if msg.get("citations"):
            with st.expander("📚 Sources", expanded=False):
                for i, c in enumerate(msg["citations"], 1):
                    title =   c.get("title", "Source")
                    url =     c.get("url", "")
                    content = c.get("content", "")
                    st.markdown(f"**{i}. {title}**")
                    if url:
                        st.markdown(f"🔗 `{url}`")
                    if content:
                        st.text(content)
                    st.markdown("---")


# =====================================================
# USER INPUT
# =====================================================
query = st.chat_input("Ask your question...")

if query:

    # Save user message
    st.session_state.chat.append({
        "role":    "user",
        "content": query
    })

    with st.chat_message("user"):
        st.write(query)

    # ---------------- CALL AGENT ----------------
    with st.spinner("Thinking..."):

        payload = {"question": query}

        if st.session_state.borrower_profile:
            payload["borrower_profile"] = st.session_state.borrower_profile

        result = loan_agent(payload)

    answer =    result.get("answer", "")
    citations = result.get("citations", [])
    risk =      result.get("risk", None)

    # Save assistant message with risk + citations for history replay
    st.session_state.chat.append({
        "role":      "assistant",
        "content":   answer,
        "risk":      risk,
        "citations": citations
    })

    # ---------------- ASSISTANT RESPONSE ----------------
    with st.chat_message("assistant"):

        # Risk scorecard (only for assessment responses)
        if risk:
            render_risk_scorecard(risk)
            st.divider()

        st.write(answer)

        # Citations
        if citations:
            with st.expander("📚 Sources", expanded=True):
                for i, c in enumerate(citations, 1):
                    title =   c.get("title", "Source")
                    url =     c.get("url", "")
                    content = c.get("content", "")
                    st.markdown(f"**{i}. {title}**")
                    if url:
                        st.markdown(f"🔗 `{url}`")
                    if content:
                        st.text(content)
                    st.markdown("---")