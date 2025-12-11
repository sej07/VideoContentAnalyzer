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
        'frame count': frame_count,
        'duration': duration
    }

