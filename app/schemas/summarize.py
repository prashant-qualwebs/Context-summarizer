from pydantic import BaseModel, Field
from typing import Optional, List, Literal

from app.config import LLMProvider


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(..., min_length=1)


class SummarizeRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., min_items=1)
    provider: Optional[LLMProvider] = Field(default="openai")


class SummarizeResponse(BaseModel):
    messages: List[ChatMessage] = Field(..., min_items=1)
