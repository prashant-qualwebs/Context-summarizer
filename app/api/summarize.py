from fastapi import APIRouter

from app.schemas.summarize import SummarizeRequest, SummarizeResponse
from app.service.summarize_service import summarize_response


router = APIRouter(tags=["summarize"])


@router.post("/compress", response_model=SummarizeResponse)
async def summarize(payload: SummarizeRequest) -> SummarizeResponse:
    assistant_response = await summarize_response(
        assistant_response=payload.assistant_response,
        provider=payload.provider,
    )
    return SummarizeResponse(
        assistant_response=assistant_response,
    )
