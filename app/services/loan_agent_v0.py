from langchain_google_genai import ChatGoogleGenerativeAI
from app.retrieval.retrieval_credit import query_credit_document


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1
)


def intent_agent(question):

    q = question.lower()

    assessment = [
        "borrower",
        "approve",
        "eligible",
        "risk",
        "profile",
        "qualify",
        "assess",
        "sanction",
        "repay",
        "credit"
    ]

    calculation = [
        "emi",
        "dti",
        "calculate",
        "interest",
        "monthly payment"
    ]

    policy = [
        "rbi",
        "guideline",
        "policy",
        "regulation",
        "what",
        "why",
        "how",
        "explain"
    ]

    if any(x in q for x in assessment):
        return "assessment"

    if any(x in q for x in calculation):
        return "calculation"

    if any(x in q for x in policy):
        return "policy"

    return "unknown"


def normalize_query(question):

    q = question.lower()

    expansions = {
        "dti": "debt to income ratio",
        "emi": "equated monthly installment",
        "ltv": "loan to value ratio",
        "cibil": "credit score"
    }

    for k, v in expansions.items():

        if k in q:
            q += f" {v}"

    return q


def retrieval_agent(question):

    query = normalize_query(question)

    docs = query_credit_document(
        query,
        k=5
    )

    context = []

    for i, doc in enumerate(docs, 1):

        meta = doc.get("metadata", {})

        context.append(
f"""
Source {i}

File:
{meta.get("source")}

Page:
{meta.get("page")}

Content:
{doc.get("content")}
"""
        )

    return docs, "\n".join(context)


def calculate_dti(profile):

    income = profile.get("monthly_income", 0)
    emis = profile.get("existing_emis", 0)

    loan = profile.get(
        "requested_loan_amount",
        0
    )

    tenure = profile.get(
        "tenure_months",
        0
    )

    if income == 0:
        return {}

    emi = 0

    if loan and tenure:

        r = .085 / 12

        emi = (

            loan * r * (1+r)**tenure

        ) / (

            (1+r)**tenure - 1

        )

    dti = ((emis + emi) / income) * 100

    return {
        "emi": round(emi, 2),
        "dti": round(dti, 2)
    }


def risk_engine(profile, dti):

    score = 10

    credit = profile.get(
        "credit_score",
        0
    )

    defaults = profile.get(
        "past_defaults",
        0
    )

    dti_score = dti.get(
        "dti",
        100
    )

    if credit < 650:
        score -= 3

    if defaults:
        score -= 2

    if dti_score > 50:
        score -= 3

    score = max(score, 1)

    if score >= 8:

        level = "LOW"
        rec = "APPROVE"

    elif score >= 5:

        level = "MEDIUM"
        rec = "CONDITIONAL APPROVE"

    else:

        level = "HIGH"
        rec = "REJECT"

    return {

        "risk_score": score,
        "risk_level": level,
        "recommendation": rec
    }


def reasoning_agent(
    question,
    intent,
    borrower_profile,
    context,
    dti,
    risk
):

    if intent == "assessment":

        prompt = f"""
You are a senior credit officer.

Use profile FIRST.
Use DTI and risk SECOND.
Use policy only if relevant.

Borrower:
{borrower_profile}

DTI:
{dti}

Risk:
{risk}

Policy:
{context}

Question:
{question}

Give:

Decision
Reasoning
Risks
Recommendation
"""

    elif intent in ["policy", "calculation"]:

        prompt = f"""
Answer ONLY using retrieved policy context.

If answer is missing in context,
say:

"I could not find this in the provided documents."

Context:

{context}

Question:

{question}
"""

    else:

        return (
            "I can answer loan, credit, "
            "borrower assessment, RBI policy, "
            "EMI and risk-related questions only."
        )

    response = llm.invoke(prompt)

    return response.content


def citation_agent(docs):

    citations = []

    for doc in docs:

        meta = doc.get(
            "metadata",
            {}
        )

        citations.append({

            "title":
            f"{meta.get('source')} "
            f"Page {meta.get('page')}",

            "content":
            doc.get(
                "content",
                ""
            )[:300]
        })

    return citations


def loan_agent(payload):

    question = payload.get(
        "question",
        ""
    )

    borrower_profile = payload.get(
        "borrower_profile",
        None
    )

    intent = intent_agent(
        question
    )

    docs = []
    context = ""

    if intent in [
        "policy",
        "calculation"
    ]:

        docs, context = retrieval_agent(
            question
        )


    dti = {}
    risk = {}

    if (
        borrower_profile
        and intent == "assessment"
    ):

        dti = calculate_dti(
            borrower_profile
        )

        risk = risk_engine(
            borrower_profile,
            dti
        )


    answer = reasoning_agent(
        question,
        intent,
        borrower_profile,
        context,
        dti,
        risk
    )

    citations = citation_agent(
        docs
    )

    print("Intent:", intent)
    print("Borrower:", borrower_profile)
    print("Docs:", len(docs))

    return {

        "answer":
        answer,

        "risk":
        risk if risk else None,

        "citations":
        citations
    }