from langchain_google_genai import (
    ChatGoogleGenerativeAI
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1
)


def generate_answer(

    question,
    intent,
    profile,
    context,
    dti,
    risk
):

    prompt = f"""
Question:
{question}

Borrower:
{profile}

DTI:
{dti}

Risk:
{risk}

Context:
{context}
"""

    return llm.invoke(
        prompt
    ).content