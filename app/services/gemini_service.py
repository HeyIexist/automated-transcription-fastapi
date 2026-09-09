import os
import json
import uuid
import httpx
from typing import Dict, Any, List, Optional
from datetime import datetime

from app.core.config import settings
from app.schemas.chat import (
    ChatMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    HealthStatus
)

class GeminiService:
    def __init__(self):
        self.default_model = settings.GEMINI_MODEL_ID
        self.api_key = settings.GEMINI_API_KEY

    def _get_api_key(self) -> str:
        return os.getenv("GEMINI_API_KEY", settings.GEMINI_API_KEY).strip()

    async def generate_chat(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        api_key = os.getenv("GEMINI_API_KEY", settings.GEMINI_API_KEY).strip()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is missing. Please set GEMINI_API_KEY in backend/.env")

        model_id = request.model or self.default_model
        if not model_id.startswith("gemini-"):
            model_id = "gemini-2.5-flash"

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}"

        # Build contents from messages
        prompt_parts = []
        if request.system_prompt:
            prompt_parts.append(f"System Instructions:\n{request.system_prompt}\n\n")

        for msg in request.messages:
            prompt_parts.append(f"{msg.role.upper()}: {msg.content}")

        full_prompt = "\n\n".join(prompt_parts)

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": full_prompt}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": request.temperature,
                "topP": request.top_p if request.top_p else 0.95,
                "responseMimeType": "application/json"
            }
        }

        req_id = f"chatcmpl-gemini-{uuid.uuid4().hex[:12]}"
        start_time = datetime.utcnow()

        async with httpx.AsyncClient(timeout=60.0) as client:
            res = await client.post(url, json=payload)
            if res.status_code != 200:
                raise RuntimeError(f"Google Gemini API error (HTTP {res.status_code}): {res.text}")

            data = res.json()
            output_text = ""
            try:
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    for p in parts:
                        if "text" in p:
                            output_text += p["text"]
            except Exception as e:
                raise RuntimeError(f"Failed to parse Gemini response: {str(e)}")

            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000.0

            return ChatCompletionResponse(
                id=req_id,
                model=model_id,
                created_at=datetime.utcnow(),
                message=ChatMessage(role="assistant", content=output_text),
                total_duration_ms=round(duration_ms, 2),
                eval_count=len(output_text.split()),
                eval_duration_ms=round(duration_ms, 2)
            )

gemini_service = GeminiService()
