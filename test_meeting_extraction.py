import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.schemas.meeting import MeetingExtractionRequest
from app.services.meeting_service import meeting_service, SAMPLE_TRANSCRIPTS
from fastapi import HTTPException


async def run_tests():
    print("==================================================")
    print(" TESTING MEETING INTELLIGENCE & ACTION ITEM EXTRACTOR")
    print("==================================================\n")

    # Test 1: Empty Transcript Edge Case
    print("[TEST 1] Testing Empty Transcript Edge Case...")
    try:
        meeting_service.validate_transcript("")
        print("[FAIL] Empty transcript was not rejected!")
    except HTTPException as e:
        print(f"[PASS] Successfully caught empty transcript with HTTP {e.status_code}: '{e.detail}'")

    # Test 2: Sample 1 - Multi-speaker action item resolution (Pricing & VPC migration)
    print("\n[TEST 2] Analyzing Sample 1 (Multi-speaker dialogues)...")
    sample_1 = SAMPLE_TRANSCRIPTS[0]
    req_1 = MeetingExtractionRequest(transcript=sample_1.transcript)
    res_1 = await meeting_service.extract_intelligence(req_1)
    print(f"Summary: {res_1.summary}")
    print(f"Key Decisions ({len(res_1.key_decisions)}): {res_1.key_decisions}")
    print(f"Action Items ({len(res_1.action_items)}):")
    for item in res_1.action_items:
        print(f"  - Owner: {item.owner} | Task: {item.task} | Due: {item.due_date}")

    assert len(res_1.action_items) >= 1, "Should extract action items for Riya / Karan"
    print("[PASS] Test 2 completed cleanly!")

    # Test 3: Sample 2 - No Action Items (Zero Hallucination)
    print("\n[TEST 3] Analyzing Sample 2 (Informational Townhall - No Action Items)...")
    sample_2 = SAMPLE_TRANSCRIPTS[1]
    req_2 = MeetingExtractionRequest(transcript=sample_2.transcript)
    res_2 = await meeting_service.extract_intelligence(req_2)
    print(f"Summary: {res_2.summary}")
    print(f"Key Decisions: {res_2.key_decisions}")
    print(f"Action Items Count: {len(res_2.action_items)}")
    print(f"Action Items: {res_2.action_items}")

    assert len(res_2.action_items) == 0, f"Expected 0 action items, got {len(res_2.action_items)}"
    print("[PASS] Test 3 (Zero Hallucination) passed cleanly!")

    # Test 4: Sample 3 - Low Confidence Dialogue Segments
    print("\n[TEST 4] Analyzing Sample 3 (Audio with inaudible noise)...")
    sample_3 = SAMPLE_TRANSCRIPTS[2]
    req_3 = MeetingExtractionRequest(transcript=sample_3.transcript)
    res_3 = await meeting_service.extract_intelligence(req_3)
    print(f"Summary: {res_3.summary}")
    print(f"Action Items Count: {len(res_3.action_items)}")
    print(f"Low Confidence Notes ({len(res_3.low_confidence_notes)}): {res_3.low_confidence_notes}")

    assert len(res_3.low_confidence_notes) > 0, "Expected low-confidence notes to be flagged!"
    print("[PASS] Test 4 passed cleanly!")

    print("\n==================================================")
    print(" ALL 4 MEETING EXTRACTION TESTS PASSED SUCCESSFULLY!")
    print("==================================================")


if __name__ == "__main__":
    asyncio.run(run_tests())
