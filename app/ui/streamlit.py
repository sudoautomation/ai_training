import os
import sys
import streamlit as st

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../../"
        )
    )
)

from app.services.loan_agent import loan_agent


st.set_page_config(
    page_title="AI Financial Assistant",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 AI Financial Assistant")

st.markdown(
"""
Ask about:

• EMI  
• DTI  
• RBI guidelines  
• Credit risk  
• Borrower eligibility  
• Loan approval
"""
)


if "chat" not in st.session_state:
    st.session_state.chat = []

if "borrower_profile" not in st.session_state:
    st.session_state.borrower_profile = None



with st.sidebar:

    st.header("📋 Borrower Profile")

    with st.expander(
        "Enter Borrower Details",
        expanded=True
    ):

        customer_id = st.text_input(
            "Customer ID"
        )

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=75,
            value=30
        )

        employment_type = st.selectbox(

            "Employment Type",

            [
                "Salaried",
                "Self-Employed",
                "Business",
                "Freelance",
                "Other"
            ]
        )

        monthly_income = st.number_input(
            "Monthly Income ₹",
            value=0,
            step=1000
        )

        existing_emis = st.number_input(
            "Existing EMI ₹",
            value=0,
            step=500
        )

        requested_loan_amount = st.number_input(
            "Requested Loan ₹",
            value=0,
            step=10000
        )

        loan_type = st.selectbox(

            "Loan Type",

            [
                "Personal Loan",
                "Home Loan",
                "Auto Loan",
                "Business Loan",
                "Education Loan"
            ]
        )

        tenure_months = st.number_input(
            "Tenure",
            value=60,
            min_value=1,
            max_value=360
        )

        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=900,
            value=700
        )

        past_defaults = st.number_input(
            "Past Defaults",
            min_value=0,
            value=0
        )

        employment_stability_years = st.number_input(
            "Employment Stability",
            value=2
        )

        bank_balance_avg = st.number_input(
            "Average Balance ₹",
            value=0,
            step=1000
        )


        c1, c2 = st.columns(2)

        with c1:

            if st.button(
                "✅ Set Profile",
                use_container_width=True
            ):

                st.session_state.borrower_profile = {

                    "customer_id":
                    customer_id,

                    "age":
                    age,

                    "employment_type":
                    employment_type,

                    "monthly_income":
                    monthly_income,

                    "existing_emis":
                    existing_emis,

                    "requested_loan_amount":
                    requested_loan_amount,

                    "loan_type":
                    loan_type,

                    "tenure_months":
                    tenure_months,

                    "credit_score":
                    credit_score,

                    "past_defaults":
                    past_defaults,

                    "employment_stability_years":
                    employment_stability_years,

                    "bank_balance_avg":
                    bank_balance_avg
                }

                st.rerun()


        with c2:

            if st.button(
                "🗑 Clear Profile",
                use_container_width=True
            ):

                st.session_state.borrower_profile = None
                st.rerun()



    if st.session_state.borrower_profile:

        p = st.session_state.borrower_profile

        st.success(

f"""
Active Profile

Customer:
{p.get("customer_id")}

Income:
₹{p.get("monthly_income"):,}

Loan:
₹{p.get("requested_loan_amount"):,}

CIBIL:
{p.get("credit_score")}
"""
        )

    else:

        st.info(
            "No borrower profile loaded"
        )


    if st.button(
        "🗑 Clear Chat",
        use_container_width=True
    ):

        st.session_state.chat = []

        st.rerun()



def render_risk(risk):

    score = risk.get(
        "risk_score",
        "-"
    )

    level = risk.get(
        "risk_level",
        "-"
    )

    recommendation = risk.get(
        "recommendation",
        "-"
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Risk Score",
            score
        )

    with c2:
        st.metric(
            "Risk Level",
            level
        )

    with c3:
        st.metric(
            "Recommendation",
            recommendation
        )



for msg in st.session_state.chat:

    with st.chat_message(
        msg["role"]
    ):

        st.write(
            msg["content"]
        )

        if msg.get("risk"):

            render_risk(
                msg["risk"]
            )

        if msg.get("citations"):

            with st.expander(
                "📚 Sources"
            ):

                for c in msg["citations"]:

                    st.markdown(
                        f"**{c.get('title')}**"
                    )

                    if c.get(
                        "content"
                    ):

                        st.text(
                            c["content"]
                        )



query = st.chat_input(
    "Ask your question..."
)


if query:

    st.session_state.chat.append(

        {
            "role":
            "user",

            "content":
            query
        }

    )


    with st.chat_message(
        "user"
    ):

        st.write(
            query
        )


    payload = {

        "question":
        query
    }


    use_profile = False


    if st.session_state.borrower_profile:

        borrower_signals = [

            "borrower",
            "customer",
            "approve",
            "eligible",
            "risk",
            "profile",
            "loan",
            "assess",
            "qualify",
            "repay",
            "credit",
            "income",
            "emi",
            "default",
            "recommend",
            "should",
            "can this",
            "will this"
        ]


        use_profile = any(

            signal in query.lower()

            for signal in borrower_signals

        )


        # short queries with profile
        if len(query.split()) <= 8:

            use_profile = True


    if (

        st.session_state.borrower_profile
        and use_profile

    ):

        payload[
            "borrower_profile"
        ] = (

            st.session_state
            .borrower_profile
        )


    with st.spinner(
        "Thinking..."
    ):

        result = loan_agent(
            payload
        )


    answer = result.get(
        "answer",
        ""
    )

    citations = result.get(
        "citations",
        []
    )

    risk = result.get(
        "risk",
        None
    )


    if not use_profile:

        risk = None


    st.session_state.chat.append(

        {

            "role":
            "assistant",

            "content":
            answer,

            "risk":
            risk,

            "citations":
            citations
        }

    )


    with st.chat_message(
        "assistant"
    ):

        if risk:

            render_risk(
                risk
            )

            st.divider()


        st.write(
            answer
        )


        if citations:

            with st.expander(

                "📚 Sources",

                expanded=True
            ):

                for i, c in enumerate(

                    citations,

                    1
                ):

                    st.markdown(

f"**{i}. {c.get('title')}**"

                    )

                    if c.get(
                        "content"
                    ):

                        st.text(

                            c[
                                "content"
                            ]
                        )