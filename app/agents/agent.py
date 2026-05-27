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


# SYSTEM PROMPT

SYSTEM_PROMPT = """
You are an AI Credit Risk Officer.

Rules:
- Use only retrieved evidence
- Never hallucinate
- Cite facts as:
  [Source: file | Page: X]
- Keep replies under 5 sentences
- Ask minimum follow-up questions
- Never expose raw chunks
- Use only one tool unless necessary
"""


# MODELS

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0
)

classifier_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0
)

# SPECIALIZED AGENTS

policy_agent = create_agent(
    model=model,
    tools=[retrieve_policy],
    system_prompt=SYSTEM_PROMPT,
    max_iterations=1
)

emi_agent = create_agent(
    model=model,
    tools=[calculate_emi],
    system_prompt=SYSTEM_PROMPT,
    max_iterations=1
)

dti_agent = create_agent(
    model=model,
    tools=[calculate_dti],
    system_prompt=SYSTEM_PROMPT,
    max_iterations=1
)

eligibility_agent = create_agent(
    model=model,
    tools=[calculate_eligibility],
    system_prompt=SYSTEM_PROMPT,
    max_iterations=1
)

documents_agent = create_agent(
    model=model,
    tools=[get_document_checklist],
    system_prompt=SYSTEM_PROMPT,
    max_iterations=1
)


# CLASSIFIER AGENT

classifier_agent = create_agent(
    model=classifier_model,
    tools=[],
    system_prompt="""
You are an intent classifier.

Classify the query into ONLY one category:

- policy
- emi
- dti
- eligibility
- documents

Rules:
- Return ONLY the category name
- No explanation
- No extra text
"""
)


# AGENT ROUTER

AGENT_MAP = {
    "policy": policy_agent,
    "emi": emi_agent,
    "dti": dti_agent,
    "eligibility": eligibility_agent,
    "documents": documents_agent
}

# RESPONSE NORMALIZER


def normalize_response(content):

    if isinstance(content, list):

        return "".join(
            item.get("text", "")
            for item in content
            if (
                isinstance(item, dict)
                and item.get("type") == "text"
            )
        ).strip()

    return str(content).strip()



# CLASSIFICATION FUNCTION

def classify_query(query: str) -> str:

    response = classifier_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": query
            }
        ]
    })

    intent = normalize_response(
        response["messages"][-1].content
    ).lower()

    return intent

# MAIN CHAT FUNCTION


def chat_with_agent(messages):

    user_query = messages[-1].content

    # STEP 1  classify intent
    intent = classify_query(user_query)

    # STEP 2  select agent
    selected_agent = AGENT_MAP.get(
        intent,
        policy_agent
    )

    # STEP 3  invoke agent
    response = selected_agent.invoke({
        "messages": messages
    })

    # STEP 4 normalize output
    final_response = normalize_response(
        response["messages"][-1].content
    )

    return final_response
