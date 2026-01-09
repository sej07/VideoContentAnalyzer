from src.video_processor import extract_frames
from src.object_detection import visualize_detections 
from src.object_tracking import ObjectTracker 

video_path = "data/sample/test_video.mp4"
output_dir = "outputs/test_frames"

print("Extracting frames...")
frame_paths = extract_frames(video_path, output_dir, sample_rate=2.0) # Increased rate for better tracking
print(f"✓ {len(frame_paths)} frames ready\n")

# Initialize the TRACKER, not the Detector
tracker = ObjectTracker(model_name='yolov8n.pt', confidence_threshold=0.5)

print("Running tracking...")
tracking_results = tracker.track_video_frames(frame_paths)

print("\nTracking Analysis:")
# Collect all unique track IDs
all_track_ids = set()
for detections in tracking_results.values():
    for det in detections:
        if det['track_id'] != -1:
            all_track_ids.add(det['track_id'])

print(f"Total unique objects tracked: {len(all_track_ids)}")
print(f"Track IDs: {sorted(list(all_track_ids))}")

# Visualization (First, Middle, Last frame)
frames_to_viz = [frame_paths[0], frame_paths[len(frame_paths)//2], frame_paths[-1]]
for i, frame in enumerate(frames_to_viz):
    output = f"outputs/tracking_frame_{i}.jpg"
    if frame in tracking_results:
        visualize_detections(frame, tracking_results[frame], output, show_track_id=True)