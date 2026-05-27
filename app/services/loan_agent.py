'''from langchain_google_genai import ChatGoogleGenerativeAI
from app.retrieval.retrieval import query_documents
from app.tools.web_search import search_web

# ---------------- LLM ----------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)

# ---------------- INTENT DETECTION ----------------
def detect_intent(q: str):

    q = q.lower()

    if any(x in q for x in ["what is", "explain", "define"]):
        return "info"

    if any(x in q for x in ["how much", "loan", "emi", "eligible", "calculate"]):
        return "calculation"

    return "general"


# ---------------- AGENT ----------------
def loan_agent(user_input: dict):

    question = user_input["question"]

    intent = detect_intent(question)

    # ---------------- ALWAYS WEB SEARCH ----------------
    web_results = search_web(question + " RBI banking India guidelines")

    # ---------------- INFO FLOW ----------------
    if intent == "info":

        docs = query_documents(question, k=5)
        context = "\n\n".join([d["content"] for d in docs])

        response = llm.invoke(f"""
You are a financial assistant.

Answer clearly using context + general knowledge.

QUESTION: {question}

CONTEXT:
{context}
""")

        return {
            "answer": response.content,
            "citations": web_results
        }

    # ---------------- CALCULATION FLOW ----------------
    if intent == "calculation":

        response = llm.invoke(f"""
You are a banking assistant.

If required information is missing, ask follow-up questions.

Otherwise provide loan eligibility reasoning.

QUESTION: {question}
""")

        return {
            "answer": response.content,
            "citations": web_results
        }

    # ---------------- GENERAL FLOW ----------------
    response = llm.invoke(f"""
You are a helpful financial assistant.

Answer the user query clearly.

QUESTION: {question}
""")

    return {
        "answer": response.content,
        "citations": web_results
    }

from langchain_google_genai import ChatGoogleGenerativeAI
from app.retrieval.retrieval import query_documents
from app.tools.web_search import search_web

# ---------------- LLM ----------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)

# ---------------- 1. INTENT AGENT ----------------
def intent_agent(question: str):
    q = question.lower()

    if any(x in q for x in ["what is", "explain", "define"]):
        return "info"

    if any(x in q for x in ["loan", "emi", "eligible", "calculate", "how much"]):
        return "calculation"

    return "general"


# ---------------- 2. RETRIEVAL AGENT (RAG + WEB) ----------------
def retrieval_agent(question: str, intent: str):

    docs = query_documents(question, k=5)
    context = "\n\n".join([d["content"] for d in docs])

    web_results = []

    # Only call web for info/calculation (avoid noise)
    if intent in ["info", "calculation"]:
        web_results = search_web(question + " RBI banking India guidelines")

    return {
        "context": context,
        "web": web_results
    }


# ---------------- 3. REASONING AGENT ----------------
def reasoning_agent(question: str, context: str, intent: str):

    if intent == "calculation":
        prompt = f"""
You are a loan risk and eligibility analyst.

Solve step by step:
- interpret user request
- check eligibility logic
- explain reasoning clearly

QUESTION: {question}

CONTEXT:
{context}
"""
    else:
        prompt = f"""
You are a financial assistant.

Answer using context clearly and accurately.

QUESTION: {question}

CONTEXT:
{context}
"""

    response = llm.invoke(prompt)
    return response.content


# ---------------- 4. FINAL FORMATTING AGENT ----------------
def final_agent(answer: str, citations: list):

    prompt = f"""
Convert the following into a clean professional banking report answer.

Keep it structured, simple, and exam/report ready.

ANSWER:
{answer}
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "citations": citations
    }


# ---------------- 5. MAIN SEQUENTIAL LOAN AGENT ----------------
def loan_agent(user_input: dict):

    question = user_input["question"]

    # Step 1: Intent
    intent = intent_agent(question)

    # Step 2: Retrieval
    data = retrieval_agent(question, intent)

    # Step 3: Reasoning
    reasoning_output = reasoning_agent(
        question,
        data["context"],
        intent
    )

    # Step 4: Final formatting
    result = final_agent(reasoning_output, data["web"])

    return result '''


'''import json
import re
from langchain_google_genai import ChatGoogleGenerativeAI

from app.retrieval.retrieval import query_documents
from app.tools.web_search import search_web


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1
)


# ---------------- INTENT ----------------
def intent_agent(q: str):

    q = q.lower()

    if any(x in q for x in ["what is", "define", "explain", "credit score", "dti"]):
        return "info"

    if any(x in q for x in ["loan", "emi", "eligible", "calculate"]):
        return "calculation"

    return "general"


# ---------------- SAFE LLM CALL ----------------
def safe_llm_json(prompt: str):
    try:
        res = llm.invoke(prompt)

        if not res or not res.content:
            return {}

        text = res.content.strip()

        # remove markdown blocks if any
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL).strip()

        return json.loads(text)

    except Exception as e:
        print("❌ JSON ERROR:", e)
        return {}


# ---------------- FINANCIAL EXTRACTION (ONLY IF NEEDED) ----------------
def extract_financial_context(question: str):

    prompt = f"""
Extract financial data ONLY if present.

Return valid JSON:
{{
  "income": null,
  "existing_emi": null,
  "loan_amount": null,
  "tenure_years": null,
  "credit_score": null
}}

Question:
{question}
"""

    return safe_llm_json(prompt)


# ---------------- RISK ENGINE ----------------
def risk_engine(data: dict):

    income = data.get("income") or 0
    emi = data.get("existing_emi") or 0
    credit = data.get("credit_score") or 0

    dti = (emi / income) * 100 if income else 100

    if credit >= 750 and dti < 40:
        risk = "LOW"
    elif credit >= 700 and dti < 50:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    return {"dti": round(dti, 2), "risk": risk}


# ---------------- RETRIEVAL ----------------
def retrieval_agent(question: str, intent: str):

    docs = query_documents(question, k=5)
    context = "\n\n".join([d["content"] for d in docs])

    web_results = []

    if intent in ["info", "calculation"]:
        web_results = search_web(question + " RBI banking India")

    return {"context": context, "web": web_results}


# ---------------- REASONING ----------------
def reasoning_agent(question, context, risk, financial):

    return llm.invoke(f"""
You are a senior loan officer.

QUESTION:
{question}

RISK:
{json.dumps(risk, indent=2)}

DATA:
{json.dumps(financial, indent=2)}

CONTEXT:
{context}

Give clear banking explanation and final decision.
""").content


# ---------------- FINAL ----------------
def format_response(answer, risk, web):

    return {
        "answer": answer,
        "risk": risk,
        "citations": web
    }


# ---------------- MAIN AGENT ----------------
def loan_agent(user_input: dict):

    question = user_input["question"]

    print("\nQUESTION:", question)

    intent = intent_agent(question)
    print("INTENT:", intent)

    
    financial_data = {}

    if intent == "calculation":
        financial_data = extract_financial_context(question)

    print("FINANCIAL DATA:", financial_data)

    risk_info = risk_engine(financial_data)
    print("RISK:", risk_info)

    retrieval = retrieval_agent(question, intent)

    answer = reasoning_agent(
        question,
        retrieval["context"],
        risk_info,
        financial_data
    )

    return format_response(answer, risk_info, retrieval["web"])'''


import json
import re

from langchain_google_genai import ChatGoogleGenerativeAI
from app.retrieval.retrieval import query_documents


# =====================================================
# LLM
# =====================================================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0,
    max_output_tokens=220
)


# =====================================================
# INTENT AGENT
# =====================================================
def intent_agent(question: str):

    q = question.lower()

    if any(x in q for x in [
        "what is",
        "explain",
        "define",
        "credit score",
        "dti",
        "policy"
    ]):
        return "info"

    if any(x in q for x in [
        "loan",
        "emi",
        "eligible",
        "calculate",
        "approval"
    ]):
        return "calculation"

    return "general"


# =====================================================
# QUERY NORMALIZATION
# Better retrieval for short queries
# =====================================================
def normalize_query(question: str):

    q = question.lower()

    expansions = {
        "dti":
        "debt to income ratio dti",

        "emi":
        "equated monthly installment emi",

        "cibil":
        "cibil credit score"
    }

    for key, value in expansions.items():

        if key in q:
            q += " " + value

    return q


# =====================================================
# RETRIEVAL AGENT
# =====================================================
def retrieval_agent(question: str):

    normalized_query = (
        normalize_query(question)
    )

    docs = query_documents(
        normalized_query,
        k=4
    )

    context_parts = []

    for i, doc in enumerate(
        docs,
        start=1
    ):

        content = doc.get(
            "content",
            ""
        ).strip()

        metadata = doc.get(
            "metadata",
            {}
        )

        page = metadata.get(
            "page",
            "Unknown"
        )

        context_parts.append(
            f"""
[Chunk {i}]
Page: {page}

{content}
"""
        )

    context = "\n".join(
        context_parts
    )

    return {
        "docs": docs,
        "context": context
    }


# =====================================================
# SAFE JSON
# =====================================================
def safe_llm_json(prompt: str):

    try:

        response = llm.invoke(
            prompt
        )

        text = (
            response.content
            .strip()
        )

        text = re.sub(
            r"```json|```",
            "",
            text
        ).strip()

        return json.loads(text)

    except Exception:
        return {}


# =====================================================
# FINANCIAL EXTRACTION
# =====================================================
def extract_financial_context(
    question: str
):

    prompt = f"""
Extract financial data from query.

Return ONLY JSON.

{{
"income": null,
"existing_emi": null,
"loan_amount": null,
"credit_score": null
}}

Question:
{question}
"""

    return safe_llm_json(
        prompt
    )


# =====================================================
# RISK ENGINE
# =====================================================
def risk_engine(data: dict):

    income = (
        data.get("income")
        or 0
    )

    emi = (
        data.get(
            "existing_emi"
        )
        or 0
    )

    credit = (
        data.get(
            "credit_score"
        )
        or 0
    )

    dti = (
        (emi / income) * 100
        if income
        else 100
    )

    if (
        credit >= 750
        and dti < 40
    ):
        risk = "LOW"

    elif (
        credit >= 700
        and dti <= 50
    ):
        risk = "MEDIUM"

    else:
        risk = "HIGH"

    return {
        "risk": risk,
        "dti": round(
            dti,
            2
        )
    }


# =====================================================
# REASONING AGENT
# NO HALLUCINATION
# =====================================================
def reasoning_agent(
    question,
    context,
    intent,
    risk=None,
    financial=None
):

    if intent == "calculation":

        prompt = f"""
You are a strict loan policy assistant.

IMPORTANT RULES:
1. Answer ONLY from context.
2. Never use outside knowledge.
3. Never hallucinate.
4. If answer not found say:
"Answer not found in loan policy document."
5. Use financial data only if available.

QUESTION:
{question}

FINANCIAL DATA:
{financial}

RISK:
{risk}

CONTEXT:
{context}
"""

    else:

        prompt = f"""
You are a strict loan policy assistant.

IMPORTANT RULES:
1. Answer ONLY from context.
2. Never use outside knowledge.
3. Never hallucinate.
4. If answer not found say:
"Answer not found in loan policy document."

QUESTION:
{question}

CONTEXT:
{context}
"""

    response = llm.invoke(
        prompt
    )

    return response.content


# =====================================================
# PDF CITATIONS
# =====================================================
def citation_agent(docs):

    citations = []
    seen = set()

    for doc in docs:

        metadata = doc.get(
            "metadata",
            {}
        )

        source = metadata.get(
            "source",
            "loan_policy.pdf"
        )

        page = metadata.get(
            "page",
            "Unknown"
        )

        key = (
            source,
            page
        )

        if key not in seen:

            seen.add(key)

            citations.append({
                "source":
                source,
                "page":
                page
            })

    return citations


# =====================================================
# MAIN AGENT
# =====================================================
def loan_agent(user_input: dict):

    question = user_input[
        "question"
    ]

    print(
        "\nQUESTION:",
        question
    )

    # ----------------
    # STEP 1
    # ----------------
    intent = intent_agent(
        question
    )

    print(
        "INTENT:",
        intent
    )

    # ----------------
    # STEP 2
    # RETRIEVAL
    # ----------------
    retrieval = (
        retrieval_agent(
            question
        )
    )

    docs = retrieval[
        "docs"
    ]

    context = retrieval[
        "context"
    ]

    if not docs:

        return {
            "answer":
            "Answer not found in loan policy document.",
            "citations":
            []
        }

    # ----------------
    # STEP 3
    # FINANCIAL DATA
    # ----------------
    financial_data = {}

    risk_info = {}

    if intent == "calculation":

        financial_data = (
            extract_financial_context(
                question
            )
        )

        risk_info = (
            risk_engine(
                financial_data
            )
        )

    # ----------------
    # STEP 4
    # ANSWER
    # ----------------
    answer = reasoning_agent(
        question,
        context,
        intent,
        risk_info,
        financial_data
    )

    # ----------------
    # STEP 5
    # PDF CITATIONS
    # ----------------
    citations = (
        citation_agent(
            docs
        )
    )

    return {
        "answer":
        answer,

        "citations":
        citations
    }