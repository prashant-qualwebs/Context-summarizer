from pydantic import BaseModel, Field
from typing import Optional

from app.config import LLMProvider


class SummarizeRequest(BaseModel):
    assistant_response: str = Field(
        ...,
        min_length=1,
        description="Full assistant response before filtering.",
    )
    provider: Optional[LLMProvider] = Field(
        default="openai",
        description="LLM provider to use: 'openai' or 'ollama'. If not specified, uses the default from settings.",
    )


class SummarizeResponse(BaseModel):
    assistant_response: str
