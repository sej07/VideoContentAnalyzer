import whisper
import os
from pathlib import Path
from typing import Dict, List, Optional 
import subprocess

def extract_audio(video_path: str, output_audio_path: str) -> str:
    print(f"Extracting audio from {video_path}")
    command = [
        'ffmpeg',
        '-i', video_path,
        '-vn',
        '-acodec', 'pcm_s16le',
        '-ar', '16000',
        '-ac', '1',
        '-y',
        output_audio_path
    ]
    result = subprocess.run(command, capture_output=True, text= True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr}")
    print(f"Audio extracted to {output_audio_path}")
    return output_audio_path

class AudioTranscriber:
    def __init__(self, model_name:str = 'base'):
        print(f"loading Whisper model: {model_name}")
        self.model = whisper.load_model(model_name)
    def transcribe(self, audio_path: str, language: Optional[str]= None) -> Dict:
        print(f"Transcribing {audio_path}")
        result = self.model.transcribe(
            audio_path,
            language = language,
            word_timestamps= True, 
            verbose= False
        )
        print("Transcription complete!")
        print(f"Detected language: {result['language']}")
        return result
    
    def format_transcript(self, result: Dict) -> str: 
        formatted = []
        formatted.append(f"Language: {result['language']}\n")
        formatted.append("\nFull Text:")
        formatted.append(result['text'])
        formatted.append("\nTimestamped Segments:")
        for segment in result['segments']:
            start = segment['start']
            end = segment['end']
            text = segment['text']
            formatted.append(f"[{start:.2f}s - {end:.2f}s]: {text}")
        return '\n'.join(formatted)
    