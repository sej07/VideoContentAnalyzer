import cv2
import numpy
import os
from pathlib import Path
from typing import Tuple, Optional, Dict

# Extract meta data from video
def get_video_info(video_path:str) -> Dict[str, float]:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"VIdeo file not found: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0

    cap.release()
    return {
        'fps': fps,
        'width': width,
        'height': height,
        'frame_count': frame_count,
        'duration': duration
    }

def extract_frames(video_path: str, output_dir: str, sample_rate: float = 1.0) -> list:
    info = get_video_info(video_path)
    fps = info["fps"]
    frame_count = info["frame_count"]
    frame_interval = fps / sample_rate
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_paths = []
    frame_idx = 0
    saved_count = 0
    print(f"Extracting frames at {sample_rate} fps")
    print(f"Frame interval: {frame_interval:.2f}")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % int(frame_interval)== 0:
            frame_filename = f"Frame_{saved_count:04d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            saved_count += 1
            print(f"Saved frame {saved_count} at index {frame_idx}")
        frame_idx += 1
    cap.release()
    print(f"Extracted {saved_count} frames to {output_dir}")
    return frame_paths
