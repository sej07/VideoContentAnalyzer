from src.video_processor import get_video_info, extract_frames
from src.object_detection import ObjectDetector, visualize_detections

video_path = "data/sample/test_video.mp4"
output_dir = "outputs/test_frames"

print("Extracting frames")
frame_paths = extract_frames(video_path, output_dir, sample_rate=1.0)
print(f"✓ {len(frame_paths)} frames ready\n")
detector = ObjectDetector(model_name='yolov8n.pt', confidence_threshold=0.5)

print("Running tracking")
tracking_results = detector.track_objects_in_frames(frame_paths)
print()

print("Tracking Analysis:")

# Collect all unique track IDs
all_track_ids = set()
for detections in tracking_results.values():
    for det in detections:
        if det['track_id'] != -1:
            all_track_ids.add(det['track_id'])

print(f"Total unique objects tracked: {len(all_track_ids)}")
print(f"Track IDs: {sorted(all_track_ids)}")
print()
frames_to_viz = [frame_paths[0], frame_paths[5], frame_paths[-1]]
for i, frame in enumerate(frames_to_viz):
    output = f"outputs/tracking_frame_{i}.jpg"
    visualize_detections(frame, tracking_results[frame], output, show_track_id=True)
    print(f"Visualized: {output}")