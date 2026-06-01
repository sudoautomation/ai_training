from typing import Literal
from pydantic import BaseModel, Field

class Citation(BaseModel):
    page: int
    text: str

class AgentResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: Literal["high","medium","low"] = "high"
