import json
import re

from langchain_google_genai import ChatGoogleGenerativeAI
from app.retrieval.retrieval import query_documents


# =====================================================
# LLM
# =====================================================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1
)


# =====================================================
# INTENT AGENT
# Classifies query into: assessment | calculation | info
# =====================================================
def intent_agent(question: str) -> str:

    q = question.lower()

    # Calculation checked first — most specific signal
    calculation_keywords = [
        "calculate", "dti", "emi", "how much", "ratio",
        "monthly payment", "debt to income", "instalment", "installment"
    ]
    if any(k in q for k in calculation_keywords):
        return "calculation"

    assessment_keywords = [
        "approve", "reject", "should we", "risk assessment",
        "creditworthy", "eligible", "recommend", "borrower profile",
        "loan application", "customer_id", "past defaults",
        "employment_stability", "bank_balance",
        "can i give", "can we give", "give a loan", "give loan",
        "should i", "this borrower", "assess", "evaluate", "review",
        "is this borrower", "qualify", "qualifies", "sanction",
        "disburse", "disbursement", "worthiness", "profile"
    ]
    if any(k in q for k in assessment_keywords):
        return "assessment"

    info_keywords = [
        "what is", "explain", "define", "policy", "rbi", "norm",
        "guideline", "rule", "regulation", "criteria", "requirement"
    ]
    if any(k in q for k in info_keywords):
        return "info"

    return "info"


# =====================================================
# QUERY NORMALIZER
# Expands abbreviations for better retrieval
# =====================================================
def normalize_query(question: str) -> str:

    q = question.lower()

    expansions = {
        "dti":   "debt to income ratio DTI",
        "emi":   "equated monthly installment EMI",
        "cibil": "CIBIL credit score",
        "ltv":   "loan to value ratio LTV",
        "npa":   "non performing asset NPA",
        "rbi":   "Reserve Bank of India RBI"
    }

    for key, value in expansions.items():
        if key in q:
            q += " " + value

    return q


# =====================================================
# RETRIEVAL AGENT
# Fetches relevant chunks from vector store
# =====================================================
def retrieval_agent(question: str, borrower_profile: dict = None, intent: str = "info") -> dict:

    query = normalize_query(question)

    if borrower_profile and intent == "assessment":
        # Build enrichment using only lowercase terms to avoid triggering
        # the keyword/FTS detector in retrieval.py (_KEYWORD_RE matches [A-Z]{2,5})
        loan_type = (borrower_profile.get("loan_type") or "").lower()
        employment_type = (borrower_profile.get("employment_type") or "").lower()

        enriched = (
            f"{query} {loan_type} {employment_type} "
            f"loan eligibility credit policy lending norms"
        )
        docs = query_documents(enriched, k=6)
    else:
        docs = query_documents(query, k=5)

    context_parts = []

    for i, doc in enumerate(docs, start=1):

        content = doc.get("content", "").strip()
        metadata = doc.get("metadata", {})
        source = metadata.get("source", "policy_document.pdf")
        page = metadata.get("page", "Unknown")

        context_parts.append(
            f"[Source {i} | File: {source} | Page: {page}]\n{content}"
        )

    context = "\n\n".join(context_parts)

    return {
        "docs": docs,
        "context": context
    }


# =====================================================
# SAFE JSON EXTRACTOR
# =====================================================
def safe_llm_json(prompt: str) -> dict:

    try:
        response = llm.invoke(prompt)
        text = response.content.strip()
        text = re.sub(r"```json|```", "", text).strip()
        return json.loads(text)

    except Exception:
        return {}


# =====================================================
# FINANCIAL EXTRACTOR
# Pulls structured financial data from natural language query
# =====================================================
def extract_financial_context(question: str, borrower_profile: dict = None) -> dict:

    if borrower_profile:
        return {
            "monthly_income":          borrower_profile.get("monthly_income"),
            "existing_emis":           borrower_profile.get("existing_emis"),
            "requested_loan_amount":   borrower_profile.get("requested_loan_amount"),
            "tenure_months":           borrower_profile.get("tenure_months"),
            "credit_score":            borrower_profile.get("credit_score"),
            "employment_type":         borrower_profile.get("employment_type"),
            "employment_stability_years": borrower_profile.get("employment_stability_years"),
            "past_defaults":           borrower_profile.get("past_defaults"),
            "bank_balance_avg":        borrower_profile.get("bank_balance_avg"),
            "loan_type":               borrower_profile.get("loan_type"),
            "age":                     borrower_profile.get("age"),
        }

    prompt = f"""
Extract financial data ONLY if explicitly mentioned in the question.
Return ONLY valid JSON. No explanation, no markdown.

{{
  "monthly_income": null,
  "existing_emis": null,
  "requested_loan_amount": null,
  "tenure_months": null,
  "credit_score": null,
  "employment_type": null,
  "past_defaults": null,
  "loan_type": null
}}

Question:
{question}
"""

    return safe_llm_json(prompt)


# =====================================================
# DTI CALCULATOR
# Computes DTI dynamically — never hardcoded
# new_emi is estimated from loan amount + tenure if not provided
# =====================================================
def calculate_dti(financial: dict) -> dict:

    monthly_income = financial.get("monthly_income") or 0
    existing_emis = financial.get("existing_emis") or 0
    requested_loan_amount = financial.get("requested_loan_amount") or 0
    tenure_months = financial.get("tenure_months") or 0

    if monthly_income == 0:
        return {
            "dti_existing": None,
            "dti_proposed": None,
            "estimated_new_emi": None,
            "note": "Monthly income not provided — DTI cannot be calculated."
        }

    # Estimate new EMI using flat interest approximation (8.5% p.a.)
    # Formula: EMI = [P × r × (1+r)^n] / [(1+r)^n - 1]
    estimated_new_emi = 0

    if requested_loan_amount > 0 and tenure_months > 0:
        annual_rate = 0.085
        r = annual_rate / 12
        n = tenure_months
        estimated_new_emi = (
            requested_loan_amount * r * (1 + r) ** n
        ) / ((1 + r) ** n - 1)
        estimated_new_emi = round(estimated_new_emi, 2)

    dti_existing = round((existing_emis / monthly_income) * 100, 2)
    total_obligations = existing_emis + estimated_new_emi
    dti_proposed = round((total_obligations / monthly_income) * 100, 2)

    return {
        "dti_existing":      dti_existing,
        "dti_proposed":      dti_proposed,
        "estimated_new_emi": estimated_new_emi,
        "total_obligations": round(total_obligations, 2),
        "monthly_income":    monthly_income
    }


# =====================================================
# RISK ENGINE
# Scores borrower risk on 1–10 scale using multiple factors
# =====================================================
def risk_engine(financial: dict, dti_data: dict) -> dict:

    credit_score =              financial.get("credit_score") or 0
    past_defaults =             financial.get("past_defaults") or 0
    employment_stability_years = financial.get("employment_stability_years") or 0
    bank_balance_avg =          financial.get("bank_balance_avg") or 0
    dti_proposed =              dti_data.get("dti_proposed") or 100

    score = 10  # start at best

    # Credit score deductions
    if credit_score >= 750:
        pass
    elif credit_score >= 700:
        score -= 1
    elif credit_score >= 650:
        score -= 2
    elif credit_score >= 600:
        score -= 3
    else:
        score -= 5

    # DTI deductions
    if dti_proposed <= 30:
        pass
    elif dti_proposed <= 40:
        score -= 1
    elif dti_proposed <= 50:
        score -= 2
    elif dti_proposed <= 60:
        score -= 3
    else:
        score -= 5

    # Past defaults
    if past_defaults == 0:
        pass
    elif past_defaults == 1:
        score -= 2
    else:
        score -= 4

    # Employment stability
    if employment_stability_years >= 3:
        pass
    elif employment_stability_years >= 1:
        score -= 1
    else:
        score -= 2

    # Bank balance buffer (should cover ≥3 months EMI)
    estimated_emi = dti_data.get("estimated_new_emi") or 0
    if estimated_emi > 0:
        buffer_months = bank_balance_avg / estimated_emi if estimated_emi else 0
        if buffer_months < 1:
            score -= 1

    score = max(1, min(10, score))

    if score >= 8:
        risk_level = "LOW"
        recommendation = "APPROVE"
        confidence = "HIGH"
    elif score >= 5:
        risk_level = "MEDIUM"
        recommendation = "CONDITIONAL APPROVE"
        confidence = "MEDIUM"
    else:
        risk_level = "HIGH"
        recommendation = "REJECT"
        confidence = "HIGH"

    return {
        "risk_score":     score,
        "risk_level":     risk_level,
        "recommendation": recommendation,
        "confidence":     confidence
    }


# =====================================================
# REASONING AGENT
# Generates structured explainable assessment
# =====================================================
def reasoning_agent(
    question: str,
    context: str,
    intent: str,
    borrower_profile: dict = None,
    financial: dict = None,
    dti_data: dict = None,
    risk: dict = None
) -> str:

    # ---- Build a clean human-readable borrower summary ----
    def fmt_profile(p: dict) -> str:
        if not p:
            return "N/A"
        return (
            f"  Customer ID       : {p.get('customer_id', 'N/A')}\n"
            f"  Age               : {p.get('age', 'N/A')}\n"
            f"  Employment Type   : {p.get('employment_type', 'N/A')}\n"
            f"  Employment Tenure : {p.get('employment_stability_years', 'N/A')} years\n"
            f"  Monthly Income    : ₹{p.get('monthly_income', 0):,}\n"
            f"  Existing EMIs     : ₹{p.get('existing_emis', 0):,} / month\n"
            f"  Loan Type         : {p.get('loan_type', 'N/A')}\n"
            f"  Requested Amount  : ₹{p.get('requested_loan_amount', 0):,}\n"
            f"  Tenure            : {p.get('tenure_months', 'N/A')} months\n"
            f"  Credit Score      : {p.get('credit_score', 'N/A')}\n"
            f"  Past Defaults     : {p.get('past_defaults', 'N/A')}\n"
            f"  Avg Bank Balance  : ₹{p.get('bank_balance_avg', 0):,}"
        )

    def fmt_dti(d: dict) -> str:
        if not d:
            return "N/A"
        note = d.get("note", "")
        if note:
            return note
        return (
            f"  Existing DTI      : {d.get('dti_existing', 'N/A')}%\n"
            f"  Proposed DTI      : {d.get('dti_proposed', 'N/A')}%\n"
            f"  Estimated New EMI : ₹{d.get('estimated_new_emi', 0):,}\n"
            f"  Total Obligations : ₹{d.get('total_obligations', 0):,} / month\n"
            f"  Monthly Income    : ₹{d.get('monthly_income', 0):,}"
        )

    def fmt_risk(r: dict) -> str:
        if not r:
            return "N/A"
        return (
            f"  Risk Score        : {r.get('risk_score', 'N/A')} / 10\n"
            f"  Risk Level        : {r.get('risk_level', 'N/A')}\n"
            f"  Recommendation    : {r.get('recommendation', 'N/A')}\n"
            f"  Confidence        : {r.get('confidence', 'N/A')}"
        )

    profile_block = fmt_profile(borrower_profile)
    dti_block     = fmt_dti(dti_data)
    risk_block    = fmt_risk(risk)

    # Build optional borrower block — only included when profile exists
    borrower_block = ""
    if borrower_profile:
        borrower_block = f"""
=== BORROWER PROFILE (loaded from session) ===
{profile_block}

=== DTI ANALYSIS (system-computed from borrower data) ===
{dti_block}

=== RISK SCORING (system-computed) ===
{risk_block}
"""

    if intent == "assessment":
        prompt = f"""
You are an experienced senior loan officer and credit risk analyst.

{borrower_block}
=== POLICY CONTEXT (retrieved from internal lending documents) ===
{context}

=== LOAN OFFICER QUESTION ===
{question}

INSTRUCTIONS:
First, think step by step about what the loan officer is really asking.
Then answer directly and specifically to that question using the borrower data above.

Your answer must:
1. Directly address the specific question asked — do not give a generic template.
2. Use the borrower numbers (income, EMIs, credit score, DTI) to reason through the answer.
3. Reference policy context where relevant — cite the page/section if available.
4. If a policy threshold is not in the retrieved context, apply standard RBI/banking norms
   and clearly label it as "standard banking norm" vs "per policy document".
5. Give a clear final recommendation with reasoning — not just "refer to policy".
6. Keep it concise and practical — a loan officer reading this should immediately know what to do.

Do NOT output a rigid 6-section template unless the question is a full credit assessment.
Shape the answer to match exactly what was asked.
"""

    elif intent == "calculation":
        prompt = f"""
You are a financial analyst specializing in loan calculations for Indian banking.

{borrower_block}
=== POLICY CONTEXT ===
{context}

=== QUESTION ===
{question}

INSTRUCTIONS:
Think about what calculation the user actually needs based on their question.
Use the borrower data if available, otherwise ask for the specific missing values.

- Show each calculation step clearly with the formula used.
- For EMI: use reducing balance formula at the applicable rate (default 8.5% p.a. if not specified).
- For DTI: Existing DTI = existing EMIs ÷ income × 100; Proposed DTI = (existing + new EMI) ÷ income × 100.
- Compare result against policy threshold from context, or standard RBI norm if not in context.
- State whether the borrower passes or fails each threshold and why.

Answer the specific calculation asked — do not show all calculations if only one was asked.
"""

    else:
        # info intent — smart retrieval-first with domain fallback
        prompt = f"""
You are a knowledgeable loan policy and banking domain expert.

{borrower_block}
=== POLICY CONTEXT (retrieved from internal documents) ===
{context}

=== QUESTION ===
{question}

INSTRUCTIONS — follow this thinking process:

STEP 1 — CHECK CONTEXT:
Read the retrieved policy context carefully. Does it directly answer the question?
If yes — answer from the context and cite the page/section.

STEP 2 — RELATE IF NOT DIRECT:
If the exact answer is not in the context, think: what concept in the context is
most related to this question? Bridge the gap — explain how the related policy
applies to what was asked. Do not say "not found" if a reasonable connection exists.

STEP 3 — DOMAIN KNOWLEDGE FALLBACK:
If the context has no relevant information at all, answer using your banking and
RBI lending domain knowledge. Clearly prefix with:
"[Based on standard banking practice]" so the user knows it is not from the PDF.

STEP 4 — BORROWER CONTEXT:
If a borrower profile is loaded, connect the answer to the borrower's specific
numbers wherever it adds value (e.g. "Based on this borrower's credit score of X...").

Keep the answer clear, practical, and specific to what was asked.
Do not give a generic answer that could apply to any question.
"""

    response = llm.invoke(prompt)
    return response.content


# =====================================================
# CITATION AGENT
# Builds structured citations from retrieved docs
# Compatible with streamlit.py (title, url, content)
# =====================================================
def citation_agent(docs: list) -> list:

    citations = []
    seen = set()

    for doc in docs:

        metadata = doc.get("metadata", {})
        source = metadata.get("source", "policy_document.pdf")
        page = metadata.get("page", "Unknown")

        key = (source, str(page))

        if key in seen:
            continue

        seen.add(key)

        filename = source.split("/")[-1] if "/" in source else source

        citations.append({
            "title":   f"{filename} — Page {page}",
            "url":     f"{source}#page={page}",
            "content": doc.get("content", "")[:300].strip()
        })

    return citations


# =====================================================
# MAIN LOAN AGENT
# Entry point — handles both plain questions and
# structured borrower profile assessments
# =====================================================
def loan_agent(user_input: dict) -> dict:

    question =        user_input.get("question", "")
    borrower_profile = user_input.get("borrower_profile", None)

    # ---- Step 1: Intent ----
    intent = intent_agent(question)

    # If profile is loaded, upgrade intent to assessment unless it is a
    # pure policy info question (no borrower context needed)
    pure_info_question = intent == "info" and not any(k in question.lower() for k in [
        "this borrower", "the borrower", "borrower", "profile", "him", "her",
        "they", "assess", "evaluate", "give", "sanction", "approve", "qualify"
    ])

    if borrower_profile and not pure_info_question:
        intent = "assessment"

    # ---- Step 2: Retrieval ----
    retrieval = retrieval_agent(question, borrower_profile, intent)
    docs =      retrieval["docs"]
    context =   retrieval["context"]

    # Soft fallback — don't hard-block; let the LLM respond with what it has
    if not docs:
        context = "No relevant policy documents were retrieved for this query."

    # ---- Step 3: Financial Extraction + DTI + Risk ----
    # Always run when profile is present so the LLM always receives
    # borrower numbers, DTI, and risk score — regardless of intent.
    financial_data = {}
    dti_data =       {}
    risk_info =      {}

    if borrower_profile or intent in ("assessment", "calculation"):

        financial_data = extract_financial_context(question, borrower_profile)

        if financial_data and any(v is not None for v in financial_data.values()):
            dti_data  = calculate_dti(financial_data)
            risk_info = risk_engine(financial_data, dti_data)

    # ---- Step 4: Reasoning ----
    answer = reasoning_agent(
        question=question,
        context=context,
        intent=intent,
        borrower_profile=borrower_profile,
        financial=financial_data if financial_data else None,
        dti_data=dti_data if dti_data else None,
        risk=risk_info if risk_info else None
    )

    # ---- Step 5: Citations ----
    citations = citation_agent(docs)

    return {
        "answer":    answer,
        "citations": citations,
        "risk":      risk_info if risk_info else None
    }