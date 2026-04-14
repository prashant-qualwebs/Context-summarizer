from enum import Enum
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv()


class LLMProvider(str, Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"


class Settings(BaseSettings):
    llm_provider: LLMProvider = Field(default=LLMProvider.OLLAMA)
    
    # OpenAI specific settings
    openai_url: str = Field(default="https://api.openai.com/v1")
    openai_api_key: str = Field(default="")
    openai_model_name: str = Field(default="gpt-4o-mini")
    
    # Ollama specific settings
    ollama_url: str = Field(default="http://localhost:11434")
    ollama_model_name: str = Field(default="qwen2.5:3b")
    
    request_timeout: float = Field(default=60.0)
    
    cors_origins: str = Field(default="*")
    cors_allow_credentials: bool = Field(default=True)
    cors_allow_methods: str = Field(default="*")
    cors_allow_headers: str = Field(default="*")

    class Config:
        env_file = ".env"
        case_sensitive = False

    def get_cors_origins(self) -> list[str]:
        if self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",")]


settings = Settings()
