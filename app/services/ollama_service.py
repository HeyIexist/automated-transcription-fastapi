import json
import uuid
from typing import AsyncGenerator, Dict, Any, List, Optional
from datetime import datetime
import httpx
from app.core.config import settings
from app.schemas.chat import (
    ChatMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    StreamChunk,
    ModelInfo,
    HealthStatus
)

PERSONA_PROMPTS: Dict[str, str] = {
    "general": (
        "You are TCS AI, an enterprise-grade artificial intelligence assistant developed for Tata Consultancy Services. "
        "Provide professional, well-structured, insightful, and helpful responses to engineers, business analysts, "
        "and project managers."
    ),
    "tcs_architect": (
        "You are a Senior Principal Enterprise Architect at TCS. Guide users on cloud architecture (AWS, Azure, GCP), "
        "microservices design, scalability, security, high availability, zero-trust patterns, and cloud migration strategies."
    ),
    "code_specialist": (
        "You are a Principal Software Engineer & Code Specialist. Provide optimized, clean, secure, and production-ready "
        "code examples in Python, Dart/Flutter, TypeScript, Java, and modern DevOps tools with detailed explanations."
    ),
    "enterprise_advisor": (
        "You are a TCS Digital Transformation and Strategy Consultant. Provide executive-ready summaries, IT strategy "
        "recommendations, digital transformation roadmaps, ROI estimates, and IT governance best practices."
    )
}


class OllamaService:
    def __init__(self, base_url: str = settings.OLLAMA_BASE_URL, default_model: str = settings.DEFAULT_MODEL):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model

    def _prepare_messages(self, request: ChatCompletionRequest) -> List[Dict[str, str]]:
        messages = []

        # Determine system prompt
        if request.system_prompt:
            system_content = request.system_prompt
        elif request.persona and request.persona in PERSONA_PROMPTS:
            system_content = PERSONA_PROMPTS[request.persona]
        else:
            system_content = settings.DEFAULT_SYSTEM_PROMPT

        messages.append({"role": "system", "content": system_content})

        for msg in request.messages:
            if msg.role != "system":
                messages.append({"role": msg.role, "content": msg.content})

        return messages

    async def check_health(self) -> HealthStatus:
        ollama_connected = False
        models: List[str] = []
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.get(f"{self.base_url}/api/tags")
                if res.status_code == 200:
                    ollama_connected = True
                    data = res.json()
                    models = [m.get("name") for m in data.get("models", []) if "name" in m]
        except Exception:
            ollama_connected = False

        return HealthStatus(
            status="healthy" if ollama_connected else "degraded",
            backend_version=settings.VERSION,
            ollama_connected=ollama_connected,
            ollama_url=self.base_url,
            active_model=self.default_model,
            models_available=models,
            timestamp=datetime.utcnow()
        )

    async def list_models(self) -> List[ModelInfo]:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.get(f"{self.base_url}/api/tags")
                if res.status_code == 200:
                    data = res.json()
                    model_list = []
                    for m in data.get("models", []):
                        details = m.get("details", {})
                        model_list.append(
                            ModelInfo(
                                name=m.get("name"),
                                size=m.get("size"),
                                modified_at=m.get("modified_at"),
                                family=details.get("family"),
                                parameter_size=details.get("parameter_size")
                            )
                        )
                    return model_list
        except Exception as e:
            # Fallback if list cannot be retrieved
            return [ModelInfo(name=self.default_model)]
        return [ModelInfo(name=self.default_model)]

    async def generate_chat(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        model = request.model or self.default_model
        messages = self._prepare_messages(request)
        req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        options: Dict[str, Any] = {
            "temperature": request.temperature,
            "top_p": request.top_p,
        }
        if request.max_tokens:
            options["num_predict"] = request.max_tokens

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": options
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()

            assistant_msg = data.get("message", {}).get("content", "")
            total_duration = data.get("total_duration", 0) / 1_000_000.0  # ns to ms
            eval_duration = data.get("eval_duration", 0) / 1_000_000.0
            eval_count = data.get("eval_count", 0)

            return ChatCompletionResponse(
                id=req_id,
                model=model,
                created_at=datetime.utcnow(),
                message=ChatMessage(role="assistant", content=assistant_msg),
                total_duration_ms=round(total_duration, 2),
                eval_count=eval_count,
                eval_duration_ms=round(eval_duration, 2)
            )

    async def stream_chat(self, request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
        model = request.model or self.default_model
        messages = self._prepare_messages(request)
        req_id = f"chatcmpl-stream-{uuid.uuid4().hex[:12]}"

        options: Dict[str, Any] = {
            "temperature": request.temperature,
            "top_p": request.top_p,
        }
        if request.max_tokens:
            options["num_predict"] = request.max_tokens

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": options
        }

        async with httpx.AsyncClient(timeout=180.0) as client:
            async with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as response:
                if response.status_code != 200:
                    error_chunk = StreamChunk(
                        id=req_id,
                        model=model,
                        content=f"Error connecting to Ollama: HTTP {response.status_code}",
                        done=True
                    )
                    yield f"data: {error_chunk.model_dump_json()}\n\n"
                    return

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk_data = json.loads(line)
                        content = chunk_data.get("message", {}).get("content", "")
                        done = chunk_data.get("done", False)

                        chunk = StreamChunk(
                            id=req_id,
                            model=model,
                            content=content,
                            done=done,
                            created_at=datetime.utcnow()
                        )
                        yield f"data: {chunk.model_dump_json()}\n\n"

                        if done:
                            yield "data: [DONE]\n\n"
                            break
                    except Exception as e:
                        continue


ollama_service = OllamaService()
