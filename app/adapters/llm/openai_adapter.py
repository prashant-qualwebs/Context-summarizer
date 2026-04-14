import httpx
from fastapi import HTTPException

from app.adapters.llm.base import LLMAdapter


class OpenAIAdapter(LLMAdapter):
    def __init__(self, base_url: str = None, model: str = None, api_key: str = None, timeout: float = None) -> None:
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.model = model or "gpt-4o-mini"
        self.api_key = api_key or ""
        self.timeout = timeout or 60.0

        if not self.api_key:
            raise ValueError("API_KEY is required for OpenAI")

    async def chat(self, messages: list[dict[str, str]]) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )

            if response.status_code == 401:
                raise HTTPException(
                    status_code=401,
                    detail="Invalid OpenAI API key. Check your API_KEY in .env file",
                )

            if response.status_code == 403:
                raise HTTPException(
                    status_code=403,
                    detail="OpenAI API access forbidden. Check your API key permissions",
                )

            if response.status_code == 429:
                raise HTTPException(
                    status_code=429,
                    detail="OpenAI API rate limit exceeded. Try again later",
                )

            if response.status_code >= 400:
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error", {}).get("message", response.text)
                    error_code = error_data.get("error", {}).get("code", "unknown")
                    
                    if error_code == "invalid_api_key":
                        raise HTTPException(
                            status_code=401,
                            detail="Invalid OpenAI API key. Check your API_KEY in .env file",
                        )
                    
                    raise HTTPException(
                        status_code=502,
                        detail=f"OpenAI API error: {error_msg}",
                    )
                except ValueError:
                    raise HTTPException(
                        status_code=502,
                        detail=f"OpenAI API error: {response.text}",
                    )

            data = response.json()
            return data["choices"][0]["message"]["content"].strip()

        except HTTPException:
            raise
        except httpx.TimeoutException:
            raise HTTPException(
                status_code=504,
                detail="OpenAI API request timed out",
            )
        except httpx.ConnectError:
            raise HTTPException(
                status_code=503,
                detail="Cannot connect to OpenAI API. Service unavailable",
            )
        except httpx.NetworkError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Network error connecting to OpenAI: {str(exc)}",
            )
        except (KeyError, IndexError, TypeError) as exc:
            raise HTTPException(
                status_code=502,
                detail="Invalid OpenAI response format",
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error with OpenAI: {str(exc)}",
            ) from exc
