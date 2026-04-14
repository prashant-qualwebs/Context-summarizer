from app.adapters.llm.base import LLMAdapter
from app.adapters.llm.ollama_adapter import OllamaAdapter
from app.adapters.llm.openai_adapter import OpenAIAdapter
from app.config import LLMProvider, settings


class LLMAdapterFactory:
    @staticmethod
    def create(provider: LLMProvider = None) -> LLMAdapter:
        # Use provided provider or fall back to settings default (Ollama)
        selected_provider = provider if provider is not None else settings.llm_provider
        
        if selected_provider == LLMProvider.OPENAI:
            return OpenAIAdapter(
                base_url=settings.openai_url,
                model=settings.openai_model_name,
                api_key=settings.openai_api_key,
                timeout=settings.request_timeout,
            )
        elif selected_provider == LLMProvider.OLLAMA:
            return OllamaAdapter(
                base_url=settings.ollama_url,
                model=settings.ollama_model_name,
                timeout=settings.request_timeout,
            )
        else:
            raise ValueError(f"Unsupported provider: {selected_provider}")
