import os
from typing import List, Union
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "Meeting Intelligence Assistant Backend"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api"

    # LLM Provider settings: "bedrock", "ollama", or "auto"
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "bedrock")

    # AWS Bedrock settings
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_SESSION_TOKEN: str = os.getenv("AWS_SESSION_TOKEN", "")
    BEDROCK_MODEL_ID: str = os.getenv("BEDROCK_MODEL_ID", "amazon.nova-micro-v1:0")

    # Ollama settings (fallback)
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "qwen2.5:3b")

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = False

    # CORS settings
    ALLOWED_ORIGINS: Union[List[str], str] = ["*"]

    # System Prompts
    DEFAULT_SYSTEM_PROMPT: str = (
        "You are an intelligent, professional Meeting Assistant. Provide accurate, clear, concise, "
        "and structured responses with summaries, key decisions, and action items."
    )

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
