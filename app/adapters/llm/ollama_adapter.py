import httpx
from fastapi import HTTPException

from app.adapters.llm.base import LLMAdapter


class OllamaAdapter(LLMAdapter):
    def __init__(self, base_url: str = None, model: str = None, timeout: float = None) -> None:
        self.base_url = (base_url or "http://localhost:11434").rstrip("/")
        self.model = model or "qwen2.5:3b"
        self.timeout = timeout or 60.0

    async def chat(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )

            if response.status_code >= 400:
                raise HTTPException(
                    status_code=502,
                    detail=f"Ollama API error: {response.text}",
                )

            data = response.json()
            return data["message"]["content"].strip()

        except httpx.TimeoutException:
            raise HTTPException(
                status_code=504,
                detail="Ollama request timed out. Check if Ollama is running",
            )
        except httpx.ConnectError:
            raise HTTPException(
                status_code=503,
                detail=f"Cannot connect to Ollama at {self.base_url}. Ensure Ollama is running",
            )
        except httpx.NetworkError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Network error connecting to Ollama: {str(exc)}",
            )
        except (KeyError, TypeError) as exc:
            raise HTTPException(
                status_code=502,
                detail="Invalid Ollama response format",
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error with Ollama: {str(exc)}",
            ) from exc
