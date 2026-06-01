import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from app.models import AgentResponse
from app.prompts import SYSTEM_PROMPT
from app.tools import retrieve_credit_risk_context

load_dotenv()

llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini")

llm = ChatOpenAI(
    model=llm_model,
    temperature=0.2,
)

agent = create_agent(
    model=llm,
    tools=[retrieve_credit_risk_context],
    system_prompt=SYSTEM_PROMPT,
    response_format=AgentResponse,
)