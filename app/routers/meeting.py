from typing import List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, status

from app.schemas.meeting import (
    MeetingExtractionRequest,
    MeetingExtractionResponse,
    SampleTranscript
)
from app.services.meeting_service import meeting_service, SAMPLE_TRANSCRIPTS
from app.services.audio_service import audio_service

router = APIRouter(prefix="/meeting", tags=["Meeting Intelligence"])


@router.post("/extract", response_model=MeetingExtractionResponse)
async def extract_meeting_intelligence(request: MeetingExtractionRequest):
    """
    Extract executive summary, key decisions, and action items (task, owner, due date)
    from a meeting transcript with strict edge-case validation and zero hallucination.
    """
    return await meeting_service.extract_intelligence(request)


@router.post("/extract-audio", response_model=MeetingExtractionResponse)
async def extract_meeting_audio(
    file: UploadFile = File(...),
    model: Optional[str] = Form(None)
):
    """
    Upload an audio recording (.wav, .mp3, .m4a, .flac, .ogg), convert to speech-to-text,
    and extract meeting summary, key decisions, and action items via local Ollama engine.
    """
    return await audio_service.process_audio_file(file, model)


@router.get("/samples", response_model=List[SampleTranscript])
async def get_sample_transcripts():
    """Retrieve pre-loaded meeting transcript samples for quick testing."""
    return SAMPLE_TRANSCRIPTS
