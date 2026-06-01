
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, field_validator

from app.agent import agent
from app.prompts import LOW_CONFIDENCE_DISCLAIMER, NOT_FOUND_ANSWER


router = APIRouter()


class QueryRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Question must not be blank.")
        return v.strip()


@router.post("/chat")
def chat(body: QueryRequest):
    try:
        result = agent.invoke({
            "messages": [{"role": "user", "content": body.question}]
        })
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )

    structured = result.get("structured_response")
    if structured is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Agent did not return a structured response.",
        )

    answer = structured.answer or NOT_FOUND_ANSWER
    if structured.confidence == "low":
        answer += LOW_CONFIDENCE_DISCLAIMER

    pages = sorted({citation.page for citation in structured.citations if citation.page is not None})

    return {
        "answer": answer,
        "citations": [{"page": citation.page, "text": citation.text} for citation in structured.citations],
        "confidence": structured.confidence,
        "pages": pages,
    }