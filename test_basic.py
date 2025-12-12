from src.video_processor import get_video_info, extract_frames
from src.object_detection import ObjectDetector, visualize_detections

video_path = 'data/sample/test_video.mp4'
print("Testing get_video_info")
info = get_video_info(video_path)
print(f"FPS: {info['fps']}")
print(f"Resolution: {info['width']}x{info['height']}")
print(f"Total Frames: {info['frame_count']}")
print(f"Duration: {info['duration']:.2f} seconds")
print("Test passed!")

print("Testing extract_frames()")
output_dir = "outputs/test_frames"
frame_paths = extract_frames(video_path, output_dir, sample_rate=1.0)
print(f"\nExtraction complete!")
print(f"Total frames extracted: {len(frame_paths)}")
print(f"Check folder: {output_dir}")

print("Testing Object Detection")
detector = ObjectDetector(model_name= 'yolov8n.pt', confidence_threshold= 0.5)
test_frame = frame_paths[0]
detections = detector.detect_objects(test_frame)
print(f"\n Detection in {test_frame}")
for i,det in enumerate(detections):
    print(f"{i+1}. {det['class']} (confidence: {det['confidence']:.2f})")
    print(f"bbox:{det['bbox']}")
print(f"Found {len(detections)} objects")

print("\n Visualizing detections")
output_viz = 'outputs/detection_visualization.jpg'
visualize_detections(test_frame, detections, output_viz)
print(f"Open {output_viz}")