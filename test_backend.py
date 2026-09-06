import asyncio
import sys
import os

# Add backend directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.config import settings
from app.services.ollama_service import ollama_service
from app.schemas.chat import ChatCompletionRequest, ChatMessage


async def main():
    print(f"[TEST] Checking Health on {settings.OLLAMA_BASE_URL}...")
    health = await ollama_service.check_health()
    print(f"[TEST] Health Status: {health.status}")
    print(f"[TEST] Ollama Connected: {health.ollama_connected}")
    print(f"[TEST] Available Models: {health.models_available}")

    if not health.ollama_connected:
        print("[TEST] Warning: Ollama service is not running. Please start 'ollama serve'")
        return

    print("\n[TEST] Testing Non-Streaming Chat Completion with model 'qwen2.5:3b'...")
    req = ChatCompletionRequest(
        model="qwen2.5:3b",
        messages=[
            ChatMessage(role="user", content="Respond in one short sentence: What is TCS AI?")
        ],
        persona="tcs_architect",
        temperature=0.3,
        max_tokens=60
    )
    res = await ollama_service.generate_chat(req)
    print(f"[TEST] Assistant Response:\n{res.message.content}")
    print(f"[TEST] Total Duration: {res.total_duration_ms}ms, Tokens: {res.eval_count}")

    print("\n[TEST] Testing Streaming Generator...")
    stream_req = ChatCompletionRequest(
        model="qwen2.5:3b",
        messages=[
            ChatMessage(role="user", content="Count from 1 to 5 separated by spaces.")
        ],
        temperature=0.1,
        max_tokens=30
    )
    chunks = []
    async for chunk in ollama_service.stream_chat(stream_req):
        chunks.append(chunk)
    print(f"[TEST] Received {len(chunks)} stream event frames successfully.")
    print("\n[SUCCESS] Backend test completed cleanly!")


if __name__ == "__main__":
    asyncio.run(main())
