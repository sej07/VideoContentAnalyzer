from ultralytics import YOLO
import cv2
import os
from typing import List, Dict, Tuple
import numpy as np

class ObjectDetector:
    def __init__(self, model_name: str = 'yolov8n.pt', confidence_threshold: float= 0.5):
        print(f"Loading YOLO model: {model_name}")
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        print(f"Model Loaded")
    
    def detect_objects(self, image_path:str) -> List[Dict]:
        results = self.model(image_path, conf = self.confidence_threshold, verbose = False)
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                detection = {
                    'class': result.names[int(box.cls[0])],
                    'confidence': float(box.conf[0]),
                    'bbox': box.xyxy[0].cpu().numpy().tolist()
                }
                detections.append(detection)
        return detections
    
    def detect_in_frame(self, frame_paths: List[str]) -> Dict[str, List[Dict]]:
        results ={}
        print(f"Running object detection on {len(frame_paths)} frames")
        for i, frame_path in enumerate(frame_paths):
            detections = self.detect_objects(frame_path)
            results[frame_path] = detections
            print(f"Frame {i+1}/{len(frame_paths)}: {len(detections)} objects detected")
        print("Object Detection completed")
        return results

def visualize_detections(image_path: str, detections: List[Dict], output_path: str) -> None:
    image = cv2.imread(image_path)
    for det in detections:
        x1, y1,x2,y2 = [int(coord) for coord in det['bbox']]
        cv2.rectangle(image, (x1,y1), (x2, y2), color = (0, 255,0), thickness = 3)
        label = f"{det['class']}{det['confidence']:.2f}"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(image, (x1,y1 - label_height- 10), (x1+ label_width, y1), (0, 255,0), -1)
        cv2.putText(image, label, (x1,y1 -5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
    cv2.imwrite(output_path, image)
    print(f"Saved Visualization to {output_path}")