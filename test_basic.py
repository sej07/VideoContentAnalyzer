from src.video_processor import get_video_info

video_path = 'data/sample/test_video.mp4'
print("Testing get_video_info")
info = get_video_info(video_path)
print(f"FPS: {info['fps']}")
print(f"Resolution: {info['width']}x{info['height']}")
print(f"Total Frames: {info['frame count']}")
print(f"Duration: {info['duration']:.2f} seconds")
print("Test passed!")