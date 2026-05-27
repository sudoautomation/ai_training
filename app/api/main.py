from fastapi import FastAPI
from pydantic import BaseModel

from langchain_core.messages import (
    HumanMessage,
    AIMessage
)

from app.agents.agent_old import chat_with_agent

app = FastAPI()


# REQUEST MODEL
class ChatRequest(BaseModel):
    message: str
    history: list = []


# RESPONSE MODEL
class ChatResponse(BaseModel):
    response: str


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    messages = []

    # convert history
    for item in request.history:

        if item["role"] == "user":
            messages.append(
                HumanMessage(content=item["content"])
            )

        else:
            messages.append(
                AIMessage(content=item["content"])
            )

    # add latest user message
    messages.append(
        HumanMessage(content=request.message)
    )

    # get response
    assistant_reply = chat_with_agent(messages)

    return ChatResponse(
        response=assistant_reply
    )