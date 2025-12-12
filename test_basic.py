from src.video_processor import get_video_info, extract_frames

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