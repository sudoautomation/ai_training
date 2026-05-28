# ui/sidebar.py
# Renders the borrower profile form in the sidebar.
# Each field is validated against allowed ranges.
# Inline error messages are shown per field.
# The Save button is enabled only when all fields are valid.

import streamlit as st

# Validation rules per field
# Each entry has min, max, and the error message to show on violation
FIELD_RULES = {
    "monthly_income": {
        "min": 1_000,
        "max": 10_000_000,
        "label": "Monthly Income",
        "error": "Monthly income must be between 1,000 and 10,000,000.",
    },
    "credit_score": {
        "min": 300,
        "max": 900,
        "label": "Credit Score",
        "error": "Credit score must be between 300 and 900.",
    },
    "requested_loan_amount": {
        "min": 1_000,
        "max": 50_000_000,
        "label": "Loan Amount",
        "error": "Loan amount must be between 1,000 and 50,000,000.",
    },
    "tenure_months": {
        "min": 6,
        "max": 360,
        "label": "Tenure (months)",
        "error": "Tenure must be between 6 and 360 months.",
    },
}


def render_sidebar():
    with st.sidebar:
        st.header("Borrower Profile")

        # Collect all field values first so we can cross-validate EMI against income
        customer_id = st.text_input("Customer ID")
        monthly_income = st.number_input("Monthly Income", min_value=0, value=0)
        existing_emi = st.number_input("Existing EMI", min_value=0, value=0)
        credit_score = st.number_input("Credit Score", min_value=0, value=700)
        loan_amount = st.number_input("Loan Amount", min_value=0, value=0)
        tenure = st.number_input("Tenure (months)", min_value=0, value=60)

        # Validate each field and collect errors
        errors = _validate_fields(
            customer_id=customer_id,
            monthly_income=monthly_income,
            existing_emi=existing_emi,
            credit_score=credit_score,
            loan_amount=loan_amount,
            tenure=tenure,
        )

        # Show inline error messages below the inputs
        for error in errors:
            st.error(error)

        # Save button is disabled when there are validation errors
        save_disabled = len(errors) > 0

        if st.button("Save Profile", disabled=save_disabled):
            st.session_state.borrower_profile = {
                "customer_id": customer_id,
                "monthly_income": monthly_income,
                "existing_emis": existing_emi,
                "credit_score": credit_score,
                "requested_loan_amount": loan_amount,
                "tenure_months": tenure,
            }
            st.toast("Profile saved successfully!")


def _validate_fields(
    customer_id: str,
    monthly_income: float,
    existing_emi: float,
    credit_score: int,
    loan_amount: float,
    tenure: int,
) -> list[str]:
    """
    Validate all profile fields and return a list of error messages.
    Empty list means all fields are valid.
    """
    errors = []

    # Customer ID must not be empty
    if not customer_id or not customer_id.strip():
        errors.append("Customer ID cannot be empty.")

    # Standard range checks using FIELD_RULES
    field_values = {
        "monthly_income": monthly_income,
        "credit_score": credit_score,
        "requested_loan_amount": loan_amount,
        "tenure_months": tenure,
    }

    for field, value in field_values.items():
        rule = FIELD_RULES[field]
        if not (rule["min"] <= value <= rule["max"]):
            errors.append(rule["error"])

    # Existing EMI cross-validation against monthly income
    # EMI cannot exceed 90% of income since that would leave nothing for living expenses
    if monthly_income > 0 and existing_emi > monthly_income * 0.9:
        errors.append(
            f"Existing EMI cannot exceed 90% of monthly income ({monthly_income * 0.9:,.0f})."
        )

    if monthly_income == 0 and existing_emi > 0:
        errors.append("Please enter monthly income before setting existing EMI.")

    return errors
