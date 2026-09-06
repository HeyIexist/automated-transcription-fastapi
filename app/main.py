import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.routers.chat import router as chat_router
from app.routers.meeting import router as meeting_router
from app.services.ollama_service import ollama_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup check
    print(f"[INFO] Initializing {settings.PROJECT_NAME} v{settings.VERSION}...")
    print(f"[INFO] Ollama Target: {settings.OLLAMA_BASE_URL} (Model: {settings.DEFAULT_MODEL})")
    health = await ollama_service.check_health()
    if health.ollama_connected:
        print(f"[INFO] Connected to Ollama! Available models: {health.models_available}")
    else:
        print("[WARN] Could not connect to Ollama server at startup. Ensure 'ollama serve' is running.")
    yield
    print("[INFO] Shutting down TCS AI Backend...")



app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Enterprise-grade FastAPI backend for TCS AI powered by local offline Ollama models.",
    lifespan=lifespan
)

# Configure CORS
origins = settings.ALLOWED_ORIGINS if isinstance(settings.ALLOWED_ORIGINS, list) else [settings.ALLOWED_ORIGINS]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if "*" in origins else origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(chat_router, prefix=settings.API_V1_STR)
app.include_router(meeting_router, prefix=settings.API_V1_STR)


@app.get("/")
async def root():
    return {
        "app": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "online",
        "docs_url": "/docs",
        "health_url": f"{settings.API_V1_STR}/health",
        "chat_url": f"{settings.API_V1_STR}/chat",
        "stream_url": f"{settings.API_V1_STR}/chat/stream"
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
