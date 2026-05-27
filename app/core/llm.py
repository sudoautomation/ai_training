import os
from dotenv import load_dotenv

from langchain_google_genai import (
    ChatGoogleGenerativeAI
)

load_dotenv()


def get_llm():

    return ChatGoogleGenerativeAI(
        model=os.getenv(
            "GOOGLE_LLM_MODEL",
            "gemini-2.5-flash"
        ),
        google_api_key=os.getenv(
            "GEMINI_API_KEY"
        ),
        temperature=0.2
    )