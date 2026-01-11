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
analyzer = SceneAnalyzer(model_name="ViT-B/32")
scenes = analyzer.analyze_scenes(frame_paths, scene_threshold=30.0)
integrator.add_scenes(scenes)
print(f"Analyzed {len(scenes)} scene(s)")
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