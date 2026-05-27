from langchain.tools import tool
from app.core.vector_store import create_vector_store

# VECTOR STORE
vectorstore = create_vector_store()


# POLICY RETRIEVAL TOOL
@tool
def retrieve_policy(query: str) -> str:
    """
    Retrieve relevant RBI guidelines, credit policy rules,
    underwriting criteria, eligibility thresholds,
    and FAQ content from the vector database.
    """

    docs = vectorstore.similarity_search(
        query=query,
        k=4
    )

    if not docs:
        return "No relevant policy documents found."

    formatted_docs = []

    for i, doc in enumerate(docs, start=1):

        source = doc.metadata.get(
            "source",
            "Unknown Source"
        )

        page = doc.metadata.get(
            "page",
            "Unknown Page"
        )

        formatted_docs.append(
            f"""
DOCUMENT {i}

{doc.page_content}

[Source: {source} | Page: {page}]
"""
        )

    return "\n\n".join(formatted_docs)



# DOCUMENT CHECKLIST TOOL
@tool
def get_document_checklist(
    loan_type: str,
    employment_type: str
) -> str:
    """
    Retrieve required loan application documents
    based on loan type and employment category.

    Use this for:
    - home loan documents
    - personal loan checklist
    - salaried applicant documents
    - self-employed borrower documents
    """

    query = (
        f"documents required for "
        f"{employment_type} {loan_type} loan"
    )

    docs = vectorstore.similarity_search(
        query=query,
        k=3
    )

    if not docs:
        return "No document checklist found."

    return "\n\n".join([
        doc.page_content
        for doc in docs
    ])


# DTI CALCULATION TOOL
@tool
def calculate_dti(
    income: float,
    existing_emis: float,
    proposed_emi: float
) -> str:
    """
    Calculate Debt-to-Income (DTI) ratio.

    Args:
        income:
            Gross monthly income in INR

        existing_emis:
            Existing monthly EMI obligations

        proposed_emi:
            Proposed new loan EMI

    Returns:
        DTI percentage and risk classification.
    """

    if income <= 0:
        return "Invalid income amount."

    total_emi = existing_emis + proposed_emi

    dti = (total_emi / income) * 100

    if dti < 35:
        risk = "Low Risk"

    elif dti <= 45:
        risk = "Moderate Risk"

    else:
        risk = "High Risk"

    return (
        f"DTI Ratio: {dti:.2f}%\n"
        f"Risk Category: {risk}\n"
        f"Total EMI Obligation: ₹{total_emi:,.0f}\n"
        f"Monthly Income: ₹{income:,.0f}\n"
        f"Policy Thresholds:\n"
        f"- <35% → Low Risk\n"
        f"- 35% to 45% → Moderate Risk\n"
        f"- >45% → High Risk"
    )


# EMI CALCULATION TOOL
@tool
def calculate_emi(
    loan_amount: float,
    annual_interest_rate: float,
    tenure_months: int
) -> str:
    """
    Calculate monthly EMI for a loan.

    Args:
        loan_amount:
            Principal loan amount

        annual_interest_rate:
            Annual interest rate in percentage

        tenure_months:
            Loan tenure in months
    """

    if tenure_months <= 0:
        return "Invalid loan tenure."

    monthly_rate = annual_interest_rate / (12 * 100)

    emi = (
        loan_amount
        * monthly_rate
        * ((1 + monthly_rate) ** tenure_months)
    ) / (
        ((1 + monthly_rate) ** tenure_months) - 1
    )

    return (
        f"Monthly EMI: ₹{emi:,.2f}\n"
        f"Loan Amount: ₹{loan_amount:,.0f}\n"
        f"Interest Rate: {annual_interest_rate}%\n"
        f"Loan Tenure: {tenure_months} months"
    )


# ELIGIBILITY TOOL
@tool
def calculate_eligibility(
    monthly_income: float,
    max_dti_percent: float,
    existing_emis: float
) -> str:
    """
    Estimate maximum eligible EMI amount
    based on DTI policy thresholds.
    """

    if monthly_income <= 0:
        return "Invalid income."

    allowed_emi = (
        (monthly_income * max_dti_percent / 100)
        - existing_emis
    )

    if allowed_emi <= 0:
        return (
            "Borrower is currently over leveraged "
            "based on DTI policy."
        )

    return (
        f"Maximum Eligible EMI: ₹{allowed_emi:,.2f}\n"
        f"Income: ₹{monthly_income:,.0f}\n"
        f"Existing EMIs: ₹{existing_emis:,.0f}\n"
        f"DTI Limit Used: {max_dti_percent}%"
    )