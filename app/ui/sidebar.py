import streamlit as st


def render_sidebar():
    with st.sidebar:
        st.header("Borrower Profile")

        profile = {
            "customer_id": st.text_input("Customer ID"),
            "monthly_income": st.number_input("Monthly Income", value=0),
            "existing_emis": st.number_input("Existing EMI", value=0),
            "credit_score": st.number_input("Credit Score", value=700),
            "requested_loan_amount": st.number_input("Loan Amount", value=0),
            "tenure_months": st.number_input("Tenure", value=60),
        }

        if st.button("Save Profile"):
            st.session_state.borrower_profile = profile
            st.toast("✅ Profile saved successfully!")