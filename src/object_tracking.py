from ultralytics import YOLO
from typing import List, Dict, Optional

class ObjectTracker:
    def __init__(self, model_name: str = 'yolov8n.pt', confidence_threshold: float = 0.5):
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold

    def track_in_frames(self, frame_paths: List[str]) -> Dict[str, List[Dict]]:
        results = {}
        print(f"Starting tracking on {len(frame_paths)} frames...")
        for i, frame_path in enumerate(frame_paths):
            tracking_results = self.model.track(
                frame_path, 
                conf=self.confidence_threshold,
                persist=True, 
                verbose=False,
                tracker="botsort.yaml" 
            )
            
            frame_detections = []
            for result in tracking_results:
                boxes = result.boxes
                for box in boxes:

                    track_id = int(box.id[0]) if box.id is not None else -1
                    
                    detection = {
                        'class': result.names[int(box.cls[0])],
                        'confidence': float(box.conf[0]),
                        'bbox': box.xyxy[0].cpu().numpy().tolist(),
                        'track_id': track_id
                    }
                    frame_detections.append(detection)
            
            results[frame_path] = frame_detections
            
            unique_ids = set(d['track_id'] for d in frame_detections if d['track_id'] != -1)
            if (i + 1) % 10 == 0:
                print(f"Tracked frame {i+1}/{len(frame_paths)} - Active Objects: {len(unique_ids)}")

        print("Object tracking complete")
        return results