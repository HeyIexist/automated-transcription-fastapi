import asyncio
import sys
import os
import wave
import struct

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import UploadFile
from app.services.audio_service import audio_service

async def test_audio():
    print("Testing AudioService process_audio_file...")
    # Create a 1-second silent WAV file for testing format handling
    filename = "test_speech.wav"
    sample_rate = 16000
    duration = 1  # second

    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        # 1 second of silence (zeroes)
        data = struct.pack('<' + ('h' * sample_rate * duration), *([0] * sample_rate * duration))
        wav_file.writeframes(data)

    with open(filename, 'rb') as f:
        content = f.read()

    os.remove(filename)

    upload_file = UploadFile(filename=filename, file=open(os.devnull, 'rb'))

    # Directly test _transcribe_audio_bytes
    print("Testing _transcribe_audio_bytes with silent audio...")
    try:
        audio_service._transcribe_audio_bytes(content, ".wav")
        print("[FAIL] Silent audio should raise unknown value error")
    except Exception as e:
        print(f"[PASS] Successfully handled silent audio error: {e}")

if __name__ == "__main__":
    asyncio.run(test_audio())
