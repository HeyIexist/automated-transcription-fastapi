from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime


class MessageRole(str):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = Field(..., description="Role of the message author")
    content: str = Field(..., description="Text content of the message")
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow, description="Timestamp of message creation")


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = Field(None, description="Ollama model name (defaults to qwen2.5:3b)")
    messages: List[ChatMessage] = Field(..., description="Conversation history list of messages")
    persona: Optional[str] = Field("general", description="Persona preset: general, tcs_architect, code_specialist, enterprise_advisor")
    system_prompt: Optional[str] = Field(None, description="Custom override for system prompt")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(0.9, ge=0.0, le=1.0, description="Top-p sampling")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    stream: Optional[bool] = Field(False, description="Whether to stream response")


class ChatCompletionResponse(BaseModel):
    id: str
    model: str
    created_at: datetime
    message: ChatMessage
    total_duration_ms: Optional[float] = None
    eval_count: Optional[int] = None
    eval_duration_ms: Optional[float] = None


class StreamChunk(BaseModel):
    id: str
    model: str
    content: str
    done: bool = False
    created_at: Optional[datetime] = None


class ModelInfo(BaseModel):
    name: str
    size: Optional[int] = None
    modified_at: Optional[str] = None
    family: Optional[str] = None
    parameter_size: Optional[str] = None


class HealthStatus(BaseModel):
    status: str
    backend_version: str
    ollama_connected: bool
    ollama_url: str
    active_model: str
    models_available: List[str]
    timestamp: datetime
