from abc import ABC, abstractmethod


class LLMAdapter(ABC):
    @abstractmethod
    async def chat(self, messages: list[dict[str, str]]) -> str:
        pass
