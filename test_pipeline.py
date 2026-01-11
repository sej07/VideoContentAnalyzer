from src.video_processor import get_video_info, extract_frames
from src.object_tracking import ObjectTracker
from src.audio_processing import extract_audio, AudioTranscriber
from src.scene_understanding import SceneAnalyzer
from src.data_integration import VideoAnalysisIntegrator
import os
video_path = "data/sample/test_video2.mp4"
output_base = "outputs/complete_analysis"
os.makedirs(output_base, exist_ok=True)

integrator = VideoAnalysisIntegrator()

print("Extracting video metadata")
video_info = get_video_info(video_path)
integrator.add_video_metadata(video_info)
print(f"Duration: {video_info['duration']:.2f}s")
print(f"Resolution: {video_info['width']}x{video_info['height']}")
print(f"FPS: {video_info['fps']:.2f}")
print()
print("Extracting frames...")
frames_dir = f"{output_base}/frames"
frame_paths = extract_frames(video_path, frames_dir, sample_rate=1.0)
print(f"Extracted {len(frame_paths)} frames")
print()
print("Running object detection and tracking")
tracker = ObjectTracker(model_name='yolov8n.pt', confidence_threshold=0.5)
tracking_results = tracker.track_in_frames(frame_paths)
integrator.add_frame_detections(frame_paths, tracking_results)
integrator.compute_tracks_summary()
print(f"Tracked {len(integrator.data['tracks'])} unique objects")
print()

print("Extracting and transcribing audio")
audio_path = f"{output_base}/audio.wav"
extract_audio(video_path, audio_path)
transcriber = AudioTranscriber(model_name='base')
transcript = transcriber.transcribe(audio_path)
integrator.add_audio_transcript(transcript)
os.remove(audio_path)  # Cleanup
print(f"Transcript: {len(transcript['text'])} characters")
print()
print("Analyzing scenes with CLIP")
analyzer  = SceneAnalyzer(model_name="ViT-B/32")
custom_prompts = [
    "a person giving a TED talk presentation on stage",
    "a speaker presenting at a conference",
    "someone giving a lecture or educational talk",
    "a person speaking in front of presentation slides",
    "a professional speaker on stage",
    "someone presenting research or ideas to an audience",
    "people in athletic wear running",  
    "a public speaking event with a single speaker"
]
key_frame = frame_paths[len(frame_paths)//2]
scene_desc = analyzer.describe_image(key_frame, prompt_options=custom_prompts)
best_desc = list(scene_desc.keys())[0]
scenes = [{
    'scene_number': 1,
    'start_frame': 0,
    'end_frame': len(frame_paths)-1,
    'key_frame_index': len(frame_paths)//2,
    'key_frame_path': key_frame,
    'description': best_desc,
    'confidence': scene_desc[best_desc]
}]
integrator.add_scenes(scenes)
print(f"Scene description: {best_desc} ({scene_desc[best_desc]:.1%})")
print()
print("Generating summary")
integrator.generate_summary()
print("Summary generated")
print()
print("Exporting results")
json_output = f"{output_base}/analysis_results.json"
integrator.export_json(json_output)
print()
print()
print("Summary:")
print(f"  Duration: {integrator.data['summary']['duration']:.1f} seconds")
print(f"  Scenes: {integrator.data['summary']['scene_count']}")
print(f"  Unique objects tracked: {integrator.data['summary']['unique_objects']}")
print(f"  Audio transcript: {len(integrator.data['audio']['full_transcript'])} chars")
print()
print(f"Full results saved to: {json_output}")
print()