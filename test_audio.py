from src.audio_processing import extract_audio, AudioTranscriber
import os

video_path = 'data/sample/test_video2.mp4'
audio_path = 'outputs/test_audio.wav'
extract_audio(video_path, audio_path)
print()

transcriber = AudioTranscriber(model_name='base')
result = transcriber.transcribe(audio_path)
print()

formatted = transcriber.format_transcript(result)
print(formatted)
print()

transcribe_path = 'outputs/transcript.txt'
with open(transcribe_path, 'w') as f:
    f.write(formatted)
print(f"Transcript saved to {transcribe_path}")

os.remove(audio_path)
print(f"Cleaned temp audio file")
