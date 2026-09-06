import os
import sys
from pydub import AudioSegment
from pydub.silence import split_on_silence
import speech_recognition as sr

def transcribe_audio(file_path):
    print(f"Loading audio: {file_path}")
    sound = AudioSegment.from_file(file_path)
    
    # Normalize audio to Mono, 16kHz, 16-bit PCM
    sound = sound.set_channels(1).set_frame_rate(16000).set_sample_width(2)
    
    # Chunk audio into 15-second segments
    chunk_length_ms = 15 * 1000  # 15 seconds
    chunks = [sound[i:i + chunk_length_ms] for i in range(0, len(sound), chunk_length_ms)]
    
    recognizer = sr.Recognizer()
    transcripts = []
    
    for idx, chunk in enumerate(chunks):
        temp_chunk_path = f"temp_chunk_{idx}.wav"
        chunk.export(temp_chunk_path, format="wav")
        
        try:
            with sr.AudioFile(temp_chunk_path) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data)
                transcripts.append(text)
                print(f"Chunk {idx+1}/{len(chunks)}: {text}")
        except sr.UnknownValueError:
            print(f"Chunk {idx+1}/{len(chunks)}: [unclear speech]")
        except Exception as e:
            print(f"Chunk {idx+1}/{len(chunks)} error: {e}")
        finally:
            if os.path.exists(temp_chunk_path):
                os.remove(temp_chunk_path)
                
    full_transcript = " ".join(transcripts)
    print("\nFull Transcript:")
    print(full_transcript)
    return full_transcript

if __name__ == "__main__":
    print("Chunked STT script ready.")
