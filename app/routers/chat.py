from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from typing import List

from app.core.config import settings
from app.schemas.chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelInfo,
    HealthStatus
)
from app.services.ollama_service import ollama_service, PERSONA_PROMPTS
from app.services.bedrock_service import bedrock_service

router = APIRouter(prefix="", tags=["Chat & AI"])


@router.get("/health", response_model=HealthStatus)
async def health_check():
    """Check health status of TCS AI Backend and active LLM connection (AWS Bedrock / Ollama)."""
    if settings.LLM_PROVIDER.lower() == "bedrock":
        return await bedrock_service.check_health()
    return await ollama_service.check_health()


@router.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List available models in active provider."""
    if settings.LLM_PROVIDER.lower() == "bedrock":
        return [
            ModelInfo(name="anthropic.claude-3-5-sonnet-20240620-v1:0", family="Claude 3.5 Sonnet"),
            ModelInfo(name="anthropic.claude-3-haiku-20240307-v1:0", family="Claude 3 Haiku"),
            ModelInfo(name="meta.llama3-70b-instruct-v1:0", family="Llama 3 70B"),
            ModelInfo(name="amazon.titan-text-express-v1", family="Titan Text")
        ]
    return await ollama_service.list_models()


@router.get("/personas")
async def list_personas():
    """List supported TCS AI personas and their default prompts."""
    return {
        "personas": [
            {
                "id": "general",
                "name": "TCS Enterprise AI",
                "description": "General enterprise assistant for agile teams and business stakeholders.",
                "system_prompt": PERSONA_PROMPTS["general"]
            },
            {
                "id": "tcs_architect",
                "name": "TCS Cloud & Solutions Architect",
                "description": "Expert in cloud architecture, distributed systems, scalability, and security.",
                "system_prompt": PERSONA_PROMPTS["tcs_architect"]
            },
            {
                "id": "code_specialist",
                "name": "TCS Senior Software Engineer",
                "description": "Clean code specialist in Python, Flutter/Dart, TypeScript, APIs & DevOps.",
                "system_prompt": PERSONA_PROMPTS["code_specialist"]
            },
            {
                "id": "enterprise_advisor",
                "name": "TCS Digital Strategy Consultant",
                "description": "Enterprise advisory, digital roadmaps, ROI estimation, and IT governance.",
                "system_prompt": PERSONA_PROMPTS["enterprise_advisor"]
            }
        ]
    }


@router.post("/chat", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    """Generate non-streaming chat completion from active LLM engine (AWS Bedrock / Ollama)."""
    try:
        if settings.LLM_PROVIDER.lower() == "bedrock":
            return await bedrock_service.generate_chat(request)
        return await ollama_service.generate_chat(request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate response from LLM engine: {str(e)}"
        )


@router.post("/chat/stream")
async def create_chat_stream(request: ChatCompletionRequest):
    """Stream token-by-token response using Server-Sent Events (SSE)."""
    try:
        stream_generator = (
            bedrock_service.stream_chat(request)
            if settings.LLM_PROVIDER.lower() == "bedrock"
            else ollama_service.stream_chat(request)
        )
        return StreamingResponse(
            stream_generator,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize stream with LLM engine: {str(e)}"
        )
