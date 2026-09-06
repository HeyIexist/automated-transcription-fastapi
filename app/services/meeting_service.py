import json
import re
from datetime import datetime
from typing import List, Dict, Any
from fastapi import HTTPException, status
import httpx

from app.core.config import settings
from app.schemas.meeting import (
    MeetingExtractionRequest,
    MeetingExtractionResponse,
    ActionItem,
    SampleTranscript
)
from app.schemas.chat import ChatCompletionRequest, ChatMessage
from app.services.bedrock_service import bedrock_service

SYSTEM_EXTRACTION_PROMPT = (
    "You are Meeting Intelligence AI, an advanced AI assistant specialized in analyzing meeting transcripts.\n"
    "Your objective is to extract structured, 100% accurate data from meeting dialogues without hallucinating.\n\n"
    "CRITICAL EXTRACTION RULES:\n"
    "1. SUMMARY: Provide a clear 2-3 sentence executive overview of the meeting topic, work updates, and outcomes.\n"
    "2. KEY DECISIONS: List all agreed decisions, resolutions, plan confirmations, and technical commitments discussed in the meeting. In standups or team syncs, agreed plans, targeted completion dates, and agreed solutions to blockers (e.g. Speaker A agreeing to check settings/fix issues for Speaker B) ARE Key Decisions. Do NOT leave key_decisions empty if agreements or resolutions were made.\n"
    "3. ACTION ITEMS: Identify distinct action items. Connect multi-turn requests (e.g., Speaker A asks Speaker B to check something, Speaker B agrees/accepts -> Owner is Speaker B).\n"
    "   - task: Clear description of what must be done.\n"
    "   - owner: Exact person assigned. If not specified, use 'Unassigned'. Never guess or invent names.\n"
    "   - due_date: Stated deadline (e.g., 'Friday', 'EOD tomorrow', '2pm today', 'Tomorrow'). If not mentioned, use 'Not specified'. Never invent dates.\n"
    "   - confidence: Numeric float (1.0 for clear dialogue).\n"
    "4. NO HALLUCINATION RULE: If NO action items exist, return an empty array `[]` for action_items. Do NOT invent action items, owners, or dates.\n"
    "5. LOW CONFIDENCE & UNCLEAR DIALOGUE: This section is ONLY for transcripts containing explicitly corrupted audio or text markers such as '[inaudible]', '[unclear]', or muffled audio dropouts. Standard English questions, polite requests ('can you check...'), or conversational responses ('Yeah, I'll do that') ARE HIGH CONFIDENCE DIALOGUE AND MUST NEVER BE FLAGGED AS LOW CONFIDENCE. If the transcript has no inaudible or corrupted markers, return an empty array `[]` for low_confidence_notes.\n\n"
    "OUTPUT FORMAT:\n"
    "You MUST respond ONLY with a valid JSON object adhering strictly to this schema:\n"
    "{\n"
    '  "summary": "Executive summary text",\n'
    '  "key_decisions": ["Decision 1", "Decision 2"],\n'
    '  "action_items": [\n'
    '    {\n'
    '      "task": "Task description",\n'
    '      "owner": "Owner Name or Unassigned",\n'
    '      "due_date": "Due Date or Not specified",\n'
    '      "confidence": 1.0\n'
    '    }\n'
    '  ],\n'
    '  "low_confidence_notes": []\n'
    "}"
)


SAMPLE_TRANSCRIPTS: List[SampleTranscript] = [
    SampleTranscript(
        id="sample_pricing",
        title="Product Pricing & Launch Sync",
        description="Multi-turn discussion with action item resolution across separate dialogue lines.",
        category="Sprint Planning",
        transcript=(
            "Arjun (Product Lead): Welcome team. Today we need to finalize the Q4 release roadmap and pricing updates.\n"
            "Riya (Marketing): I've benchmarked competitor pricing. Our enterprise tier should stay at $49/user/month.\n"
            "Arjun: Agreed. Let's make that official—enterprise tier remains $49.\n"
            "Riya, can you send the updated pricing sheet to the sales enablement group by Friday?\n"
            "Riya: Yes, I'll do that by Friday EOD.\n"
            "Karan (Engineering Lead): On tech debt, we must migrate the database cluster to the new cloud VPC before launch.\n"
            "Arjun: Karan, please lead the VPC migration and ensure dry runs complete by next Tuesday.\n"
            "Karan: Sure, I will take ownership of the VPC migration and have the dry run done by Tuesday."
        )
    ),
    SampleTranscript(
        id="sample_no_actions",
        title="Quarterly Tech Update (Informational)",
        description="Purely informational status update with key decisions but ZERO action items.",
        category="Townhall",
        transcript=(
            "Vikram: Good morning everyone. Today is an informational check-in on Q3 infrastructure achievements.\n"
            "Neha: System uptime for Q3 reached 99.98%, exceeding our SLA target of 99.9%.\n"
            "Vikram: That is fantastic news. We have decided to lock in AWS as our primary cloud host for 2026.\n"
            "Neha: Also, the security audit concluded with zero critical vulnerabilities reported.\n"
            "Vikram: Excellent work team. No further tasks assigned today. Meeting adjourned."
        )
    ),
    SampleTranscript(
        id="sample_audio_unclear",
        title="Client Onboarding Sync (Audio with Noise)",
        description="Transcript derived from noisy audio containing inaudible segments and low-confidence notes.",
        category="Client Meeting",
        transcript=(
            "Sarah: Let me confirm the onboarding timeline for Client X.\n"
            "David: [inaudible background noise] ... we need to setup the SSO integration ... [unclear]\n"
            "Sarah: David, can you configure SAML SSO authentication by Wednesday?\n"
            "David: Yes, I'll configure SAML SSO by Wednesday.\n"
            "Sarah: Who is preparing the user training deck?\n"
            "David: [inaudible] ... maybe someone from Customer Success ... [unclear]."
        )
    )
]


class MeetingService:
    def __init__(self, base_url: str = settings.OLLAMA_BASE_URL, default_model: str = settings.DEFAULT_MODEL):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model

    def validate_transcript(self, transcript: str) -> None:
        """Reject empty or whitespace-only transcripts immediately."""
        if not transcript or not transcript.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Transcript input cannot be empty. Please paste or upload a valid meeting transcript."
            )
        if len(transcript.strip()) < 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Transcript input is too short. Please provide a full meeting transcript."
            )

    async def _call_ollama_fallback(self, model: str, user_prompt: str) -> str:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_EXTRACTION_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": {"temperature": 0.1, "top_p": 0.9}
        }
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                res = await client.post(f"{self.base_url}/api/chat", json=payload)
                res.raise_for_status()
                data = res.json()
                return data.get("message", {}).get("content", "")
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error calling local LLM fallback: {str(e)}"
            )

    async def extract_intelligence(self, request: MeetingExtractionRequest) -> MeetingExtractionResponse:
        self.validate_transcript(request.transcript)

        transcript_text = request.transcript.strip()
        words = len(transcript_text.split())
        model = request.model or self.default_model

        user_prompt = f"Analyze the following meeting transcript:\n\n---\n{transcript_text}\n---"
        raw_content = ""
        provider = settings.LLM_PROVIDER.lower()

        # Step: Execute LLM Call (AWS Bedrock or Ollama fallback)
        if provider == "bedrock" or (provider == "auto" and settings.AWS_ACCESS_KEY_ID):
            try:
                bedrock_req = ChatCompletionRequest(
                    model=request.model or settings.BEDROCK_MODEL_ID,
                    messages=[ChatMessage(role="user", content=user_prompt)],
                    system_prompt=SYSTEM_EXTRACTION_PROMPT,
                    temperature=0.1,
                    top_p=0.9
                )
                bedrock_res = await bedrock_service.generate_chat(bedrock_req)
                raw_content = bedrock_res.message.content
            except Exception as e:
                err_msg = str(e)
                if "credentials" in err_msg.lower() or provider == "auto":
                    try:
                        raw_content = await self._call_ollama_fallback(model, user_prompt)
                    except Exception:
                        raise HTTPException(
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=(
                                "AWS Bedrock credentials missing or invalid. "
                                "Please add AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_REGION to backend/.env, "
                                "or ensure local Ollama is running."
                            )
                        )
                else:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"AWS Bedrock LLM engine error: {err_msg}"
                    )
        else:
            raw_content = await self._call_ollama_fallback(model, user_prompt)

        # Parse Structured JSON Response
        parsed = self._parse_json_response(raw_content)

        # Construct Response Object
        action_items = []
        for item in parsed.get("action_items", []):
            if isinstance(item, dict) and item.get("task"):
                action_items.append(
                    ActionItem(
                        task=str(item.get("task")).strip(),
                        owner=str(item.get("owner", "Unassigned")).strip(),
                        due_date=str(item.get("due_date", "Not specified")).strip(),
                        confidence=float(item.get("confidence", 1.0)),
                        is_completed=False
                    )
                )

        # Ensure low-confidence notes are only present if explicit noise markers exist in transcript
        raw_lc_notes = parsed.get("low_confidence_notes", [])
        low_confidence_notes: List[str] = []
        has_noise = any(marker in transcript_text.lower() for marker in ["[inaudible]", "[unclear]", "[muffled]", "[noise]", "[garbled]"])

        if has_noise:
            if raw_lc_notes:
                low_confidence_notes = [str(n) for n in raw_lc_notes]
            else:
                low_confidence_notes.append("Detected inaudible or unclear audio segments in transcript dialogue.")

        return MeetingExtractionResponse(
            summary=parsed.get("summary", "Summary not available."),
            key_decisions=parsed.get("key_decisions", []),
            action_items=action_items,
            low_confidence_notes=low_confidence_notes,
            processed_at=datetime.utcnow(),
            word_count=words
        )

    def _parse_json_response(self, raw_text: str) -> Dict[str, Any]:
        """Extracts JSON from raw LLM output, handling markdown blocks."""
        cleaned = raw_text.strip()
        if "```" in cleaned:
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
            if match:
                cleaned = match.group(1)
            else:
                cleaned = re.sub(r"^```(?:json)?", "", cleaned)
                cleaned = re.sub(r"```$", "", cleaned).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass

        return {
            "summary": raw_text[:300] if raw_text else "Extraction completed.",
            "key_decisions": [],
            "action_items": [],
            "low_confidence_notes": ["Raw model output could not be strictly parsed as JSON."]
        }


meeting_service = MeetingService()
