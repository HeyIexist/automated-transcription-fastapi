import os
import tempfile
import io
from typing import Optional
from fastapi import HTTPException, UploadFile, status
import speech_recognition as sr
from pydub import AudioSegment

from app.schemas.meeting import MeetingExtractionRequest, MeetingExtractionResponse
from app.services.meeting_service import meeting_service


class AudioService:
    def __init__(self):
        self.recognizer = sr.Recognizer()

    async def process_audio_file(self, file: UploadFile, model: Optional[str] = None) -> MeetingExtractionResponse:
        filename = file.filename or "audio_input.wav"
        ext = os.path.splitext(filename)[1].lower()

        # Step 1: Validate file presence
        contents = await file.read()
        if not contents or len(contents) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded audio file is empty. Please select a valid meeting recording."
            )

        # Supported audio formats
        supported_exts = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac", ".wma"}
        if ext not in supported_exts:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported audio format '{ext}'. Supported formats: {', '.join(supported_exts)}"
            )

        # Step 2: Transcribe Speech to Text with chunking & audio normalization
        transcript, is_low_confidence = self._transcribe_audio_bytes(contents, ext)

        if not transcript or not transcript.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not detect clear speech in the uploaded audio recording. Please verify the audio file contains audible spoken dialogue."
            )

        # Step 3: Send transcribed text to Ollama Meeting Intelligence Service
        request = MeetingExtractionRequest(transcript=transcript, model=model)
        extraction_res = await meeting_service.extract_intelligence(request)

        # Step 4: Attach raw transcribed text so UI can display what was extracted from audio
        extraction_res.transcribed_text = transcript

        # Step 5: Inject low confidence audio flags if detected
        if is_low_confidence:
            note = "Audio quality issue: Inaudible or muffled segments detected during speech-to-text transcription."
            if note not in extraction_res.low_confidence_notes:
                extraction_res.low_confidence_notes.append(note)

        return extraction_res

    def _transcribe_audio_bytes(self, contents: bytes, ext: str) -> tuple[str, bool]:
        """Normalizes audio to 16kHz Mono PCM WAV, splits into 12s chunks, and transcribes via SpeechRecognition."""
        is_low_confidence = False
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temp_in:
            temp_in.write(contents)
            temp_in_path = temp_in.name

        try:
            # Load audio using pydub
            try:
                sound = AudioSegment.from_file(temp_in_path)
            except Exception as pydub_err:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Could not parse audio format '{ext}'. Error: {str(pydub_err)}. Please upload a valid audio recording."
                )

            # Normalize audio: Mono channel (1), 16000 Hz frame rate, 16-bit PCM (sample_width=2)
            sound = sound.set_channels(1).set_frame_rate(16000).set_sample_width(2)

            # Split into 12-second chunks to avoid Google API single-request payload limits (HTTP 400 Bad Request)
            chunk_length_ms = 12 * 1000  # 12 seconds
            chunks = [sound[i:i + chunk_length_ms] for i in range(0, len(sound), chunk_length_ms)]

            recognized_parts = []

            for idx, chunk in enumerate(chunks):
                # Skip silent or near-silent chunks (< 250ms or very low volume)
                if len(chunk) < 250 or chunk.dBFS < -60:
                    continue

                chunk_wav_path = temp_in_path + f"_chunk_{idx}.wav"
                try:
                    chunk.export(chunk_wav_path, format="wav")
                    with sr.AudioFile(chunk_wav_path) as source:
                        audio_data = self.recognizer.record(source)
                        try:
                            text = self.recognizer.recognize_google(audio_data)
                            if text and text.strip():
                                recognized_parts.append(text.strip())
                        except sr.UnknownValueError:
                            is_low_confidence = True
                        except sr.RequestError:
                            is_low_confidence = True
                finally:
                    if os.path.exists(chunk_wav_path):
                        try:
                            os.remove(chunk_wav_path)
                        except Exception:
                            pass

            if not recognized_parts:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Google Speech Recognition could not understand the audio. Please check that your recording contains clear spoken voice."
                )

            full_transcript = " ".join(recognized_parts)
            return full_transcript, is_low_confidence

        except HTTPException:
            raise
        except Exception as err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Error processing audio file: {str(err)}"
            )
        finally:
            if os.path.exists(temp_in_path):
                try:
                    os.remove(temp_in_path)
                except Exception:
                    pass


audio_service = AudioService()
