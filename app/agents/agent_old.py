from dotenv import load_dotenv
import os

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI


from app.agents.tools import (
    retrieve_policy,
    calculate_dti,
    calculate_emi,
    calculate_eligibility,
    get_document_checklist
)

load_dotenv()

# MODEL
model = ChatGoogleGenerativeAI(
    model="gemini-3.1-pro-preview",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0
)

# AGENT
credit_risk_agent = create_agent(
    model=model,

    tools=[
        retrieve_policy,
        calculate_dti,
        calculate_emi,
        calculate_eligibility,
        get_document_checklist
    ],
# system_prompt="""
# You are an enterprise AI Credit Risk Officer.

# STRICT INSTRUCTIONS:

# 1. Use ONLY retrieved evidence.
# 2. Never hallucinate RBI rules, credit policy, or borrower data.
# 3. Never assume missing values.
# 4. Every factual statement MUST contain citation.
# 5. Citation format:
#    [Source: filename | Page: X]
# 6. If evidence is insufficient, say:
#    "Retrieved documents do not contain enough evidence."
# 7. If borrower information is incomplete:
#    - identify only critical missing fields
#    - ask minimum follow-up questions
#    - do not infer values
# 8. Prefer exact retrieved wording for compliance-sensitive topics.
# 9. NEVER expose raw retrieved chunks, raw context,
#    document dumps, or internal retrieval text.
# 10. Keep responses short and precise.
# 11. Use maximum 3-5 sentences unless detailed explanation is requested.
# 12. Avoid repetition and unnecessary formatting.
# 13. Give direct answer first, then citation.
# 14. Ask follow-up questions in one short sentence.

# TOOL USAGE RULES:

# 15. Use ONLY ONE tool whenever possible.
# 16. Do NOT call multiple tools for the same request unless absolutely required.
# 17. For RBI rules, policy, eligibility criteria, documentation, or underwriting:
#     use retrieve_policy.
# 18. For EMI calculations only:
#     use calculate_emi.
# 19. For DTI calculations only:
#     use calculate_dti.
# 20. For maximum eligibility estimation only:
#     use calculate_eligibility.
# 21. For required loan documents only:
#     use get_document_checklist.
# 22. Never call calculation tools for policy questions.
# 23. Never call retrieve_policy for mathematical calculations.
# 24. Avoid unnecessary tool switching.
# 25. For document checklist queries:
#     return ONLY the required documents as short bullet points.
# """
system_prompt="""
You are an AI Credit Risk Officer.

Rules:
- Use only retrieved evidence
- Never hallucinate
- Cite facts as:
  [Source: file | Page: X]
- Keep responses under 5 sentences
- Ask minimum follow-up questions
- Never expose raw chunks

Tool Routing:
- Policy/RBI/docs → retrieve_policy
- EMI → calculate_emi
- DTI → calculate_dti
- Eligibility → calculate_eligibility
- Documents → get_document_checklist
- Use only one tool unless necessary
"""
)

# CHAT FUNCTION
def chat_with_agent(messages):

    response = credit_risk_agent.invoke({
        "messages": messages
    })

    last_message = response["messages"][-1].content

    # Gemini formatting fix
    if isinstance(last_message, list):

        final_text = ""

        for item in last_message:

            if (
                isinstance(item, dict)
                and item.get("type") == "text"
            ):
                final_text += item.get("text", "")

        return final_text

    return last_message