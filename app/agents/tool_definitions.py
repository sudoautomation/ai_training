# agents/tool_definitions.py
# Defines Gemini-compatible function declarations for the loan agent tools.
# Each tool maps directly to one service. No business logic lives here.

TOOL_DEFINITIONS = [
    {
        "name": "search_policy",
        "description": (
            "Search the loan policy knowledge base for RBI guidelines, "
            "eligibility criteria, DTI thresholds, LTV limits, NPA rules, "
            "and any other credit policy information. Use this when the question "
            "involves policy, guidelines, how something works, or why a rule exists."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up in the policy documents."
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "calculate_dti",
        "description": (
            "Calculate the Debt-to-Income ratio and estimated EMI for a borrower. "
            "Use this when you need to compute financial figures from the borrower profile."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "monthly_income": {
                    "type": "number",
                    "description": "Borrower monthly income in rupees."
                },
                "existing_emis": {
                    "type": "number",
                    "description": "Total existing EMI obligations per month."
                },
                "requested_loan_amount": {
                    "type": "number",
                    "description": "The loan amount being requested."
                },
                "tenure_months": {
                    "type": "integer",
                    "description": "Loan repayment tenure in months."
                }
            },
            "required": ["monthly_income"]
        }
    },
    {
        "name": "evaluate_risk",
        "description": (
            "Evaluate the credit risk of a borrower and return a risk score, "
            "risk level, and approval recommendation. Use this when the question "
            "involves borrower assessment, approval eligibility, or credit risk."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "credit_score": {
                    "type": "integer",
                    "description": "Borrower CIBIL or credit score."
                },
                "past_defaults": {
                    "type": "integer",
                    "description": "Number of past loan defaults. 0 means no defaults."
                },
                "dti": {
                    "type": "number",
                    "description": "Debt-to-income ratio as a percentage, from calculate_dti."
                }
            },
            "required": ["credit_score"]
        }
    }
]
