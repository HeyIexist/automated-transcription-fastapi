from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class ActionItem(BaseModel):
    task: str = Field(..., description="Actionable task description")
    owner: str = Field("Unassigned", description="Name of person assigned, or Unassigned if not stated")
    due_date: str = Field("Not specified", description="Deadline or due date, or Not specified if not stated")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Confidence score for extracting this action item")
    is_completed: bool = Field(False, description="Completion status flag for UI")


class MeetingExtractionRequest(BaseModel):
    transcript: str = Field(..., description="Full text transcript of the meeting")
    model: Optional[str] = Field(None, description="Model to use for extraction")


class MeetingExtractionResponse(BaseModel):
    summary: str = Field(..., description="Executive summary of the meeting")
    key_decisions: List[str] = Field(default_factory=list, description="Key decisions agreed upon")
    action_items: List[ActionItem] = Field(default_factory=list, description="Structured action items")
    low_confidence_notes: List[str] = Field(default_factory=list, description="Flagged low confidence dialogue segments")
    transcribed_text: Optional[str] = Field(None, description="Raw transcribed text from uploaded audio file")
    processed_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of processing")
    word_count: int = Field(0, description="Word count of analyzed transcript")


class SampleTranscript(BaseModel):
    id: str
    title: str
    description: str
    category: str
    transcript: str
