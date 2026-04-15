from fastapi import APIRouter

from app.schemas.summarize import SummarizeRequest, SummarizeResponse
from app.service.summarize_service import summarize_response


router = APIRouter(tags=["summarize"])


@router.post("/compress", response_model=SummarizeResponse)
async def summarize(payload: SummarizeRequest) -> SummarizeResponse:
    messages = await summarize_response(
        chats=payload.messages,
        provider=payload.provider,
    )
    return SummarizeResponse(
        messages=messages,
    )
