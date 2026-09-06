import json
import uuid
import re
from typing import AsyncGenerator, Dict, Any, List, Optional
from datetime import datetime
import boto3
from botocore.exceptions import BotoCoreError, ClientError

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


class BedrockService:
    def __init__(self):
        self.region = settings.AWS_REGION
        self.default_model = settings.BEDROCK_MODEL_ID
        self._client = None

    def _get_client(self):
        kwargs: Dict[str, Any] = {"region_name": settings.AWS_REGION}
        if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
            kwargs["aws_access_key_id"] = settings.AWS_ACCESS_KEY_ID.strip()
            kwargs["aws_secret_access_key"] = settings.AWS_SECRET_ACCESS_KEY.strip()
            if settings.AWS_SESSION_TOKEN:
                kwargs["aws_session_token"] = settings.AWS_SESSION_TOKEN.strip()

        try:
            return boto3.client("bedrock-runtime", **kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize AWS Bedrock client: {str(e)}")

    def _prepare_bedrock_payload(self, request: ChatCompletionRequest):
        # Determine system prompt
        if request.system_prompt:
            system_content = request.system_prompt
        elif request.persona and request.persona in PERSONA_PROMPTS:
            system_content = PERSONA_PROMPTS[request.persona]
        else:
            system_content = settings.DEFAULT_SYSTEM_PROMPT

        system_prompts = [{"text": system_content}]

        # Prepare messages in Bedrock Converse API format
        messages = []
        for msg in request.messages:
            if msg.role == "system":
                continue
            role = "user" if msg.role == "user" else "assistant"
            messages.append({
                "role": role,
                "content": [{"text": msg.content}]
            })

        inference_config = {
            "temperature": request.temperature,
            "topP": request.top_p,
        }
        if request.max_tokens:
            inference_config["maxTokens"] = request.max_tokens

        model_id = request.model or self.default_model
        return model_id, system_prompts, messages, inference_config

    async def check_health(self) -> HealthStatus:
        bedrock_connected = False
        try:
            client = self._get_client()
            # Test ping via empty request or client check
            if client is not None:
                bedrock_connected = True
        except Exception:
            bedrock_connected = False

        return HealthStatus(
            status="healthy" if bedrock_connected else "degraded",
            backend_version=settings.VERSION,
            ollama_connected=bedrock_connected,
            ollama_url=f"AWS Bedrock ({self.region})",
            active_model=self.default_model,
            models_available=[
                "anthropic.claude-3-5-sonnet-20240620-v1:0",
                "anthropic.claude-3-haiku-20240307-v1:0",
                "meta.llama3-70b-instruct-v1:0",
                "amazon.titan-text-express-v1"
            ],
            timestamp=datetime.utcnow()
        )

    async def generate_chat(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        client = self._get_client()
        model_id, system_prompts, messages, inference_config = self._prepare_bedrock_payload(request)
        req_id = f"chatcmpl-bedrock-{uuid.uuid4().hex[:12]}"
        start_time = datetime.utcnow()

        try:
            response = client.converse(
                modelId=model_id,
                messages=messages,
                system=system_prompts,
                inferenceConfig=inference_config
            )

            output_text = ""
            output_msg = response.get("output", {}).get("message", {})
            for content_block in output_msg.get("content", []):
                if "text" in content_block:
                    output_text += content_block["text"]

            usage = response.get("usage", {})
            eval_count = usage.get("outputTokens", 0)
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000.0

            return ChatCompletionResponse(
                id=req_id,
                model=model_id,
                created_at=datetime.utcnow(),
                message=ChatMessage(role="assistant", content=output_text),
                total_duration_ms=round(duration_ms, 2),
                eval_count=eval_count,
                eval_duration_ms=round(duration_ms, 2)
            )
        except (BotoCoreError, ClientError) as e:
            raise RuntimeError(f"AWS Bedrock Converse API error: {str(e)}")

    async def stream_chat(self, request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
        client = self._get_client()
        model_id, system_prompts, messages, inference_config = self._prepare_bedrock_payload(request)
        req_id = f"chatcmpl-bedrock-stream-{uuid.uuid4().hex[:12]}"

        try:
            response = client.converse_stream(
                modelId=model_id,
                messages=messages,
                system=system_prompts,
                inferenceConfig=inference_config
            )

            stream = response.get("stream")
            if stream:
                for event in stream:
                    if "contentBlockDelta" in event:
                        delta = event["contentBlockDelta"].get("delta", {})
                        if "text" in delta:
                            text_chunk = delta["text"]
                            chunk = StreamChunk(
                                id=req_id,
                                model=model_id,
                                content=text_chunk,
                                done=False,
                                created_at=datetime.utcnow()
                            )
                            yield f"data: {chunk.model_dump_json()}\n\n"

                # Final completion frame
                final_chunk = StreamChunk(
                    id=req_id,
                    model=model_id,
                    content="",
                    done=True,
                    created_at=datetime.utcnow()
                )
                yield f"data: {final_chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
        except (BotoCoreError, ClientError) as e:
            error_chunk = StreamChunk(
                id=req_id,
                model=model_id,
                content=f"AWS Bedrock Streaming Error: {str(e)}",
                done=True
            )
            yield f"data: {error_chunk.model_dump_json()}\n\n"


bedrock_service = BedrockService()
