import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime


class VideoAnalysisIntegrator:
    def __init__(self):
        self.data = {
            "video_metadata": {},
            "audio": {},
            "scenes": [],
            "frames": [],
            "tracks": {},
            "summary": {}
        }
    def add_video_metadata(self, info: Dict) -> None:
        self.data["video_metadata"] = {
            "fps": info['fps'],
            "width": info['width'],
            "height": info['height'],
            "frame_count": info['frame_count'],
            "duration": info['duration'],
            "resolution": f"{info['width']}x{info['height']}"
        }
    def add_audio_transcript(self, transcript_result: Dict) -> None:
        self.data["audio"] = {
            "language": transcript_result.get('language', 'unknown'),
            "full_transcript": transcript_result['text'],
            "segments": []
        }
        for segment in transcript_result.get('segments', []):
            self.data["audio"]["segments"].append({
                "start": segment['start'],
                "end": segment['end'],
                "text": segment['text'].strip()
            })
    def add_scenes(self, scenes: List[Dict]) -> None:
        fps = self.data["video_metadata"].get("fps", 30)
        for scene in scenes:
            self.data["scenes"].append({
                "scene_number": scene['scene_number'],
                "start_frame": scene['start_frame'],
                "end_frame": scene['end_frame'],
                "start_time": scene['start_frame'] / fps,
                "end_time": scene['end_frame'] / fps,
                "description": scene['description'],
                "confidence": scene['confidence'],
                "key_frame_path": scene['key_frame_path']
            })
    
    def add_frame_detections(self, frame_paths: List[str], detections: Dict[str, List[Dict]]) -> None:
        fps = self.data["video_metadata"].get("fps", 30)
        for frame_idx, frame_path in enumerate(frame_paths):
            frame_detections = detections.get(frame_path, [])
            self.data["frames"].append({
                "frame_index": frame_idx,
                "timestamp": frame_idx / fps,
                "frame_path": frame_path,
                "detections": frame_detections
            })
    def compute_tracks_summary(self) -> None:
        fps = self.data["video_metadata"].get("fps", 30)
        tracks = {}
        # Aggregate data across frames
        for frame in self.data["frames"]:
            for det in frame["detections"]:
                track_id = det.get("track_id", -1)
                if track_id == -1:
                    continue
                if track_id not in tracks:
                    tracks[track_id] = {
                        "class": det["class"],
                        "first_frame": frame["frame_index"],
                        "last_frame": frame["frame_index"],
                        "confidences": [],
                        "frame_count": 0
                    }
                tracks[track_id]["last_frame"] = frame["frame_index"]
                tracks[track_id]["confidences"].append(det["confidence"])
                tracks[track_id]["frame_count"] += 1
        for track_id, data in tracks.items():
            self.data["tracks"][str(track_id)] = {
                "class": data["class"],
                "first_appearance": data["first_frame"] / fps,
                "last_appearance": data["last_frame"] / fps,
                "duration": (data["last_frame"] - data["first_frame"]) / fps,
                "total_frames": data["frame_count"],
                "avg_confidence": sum(data["confidences"]) / len(data["confidences"])
            }
    def generate_summary(self) -> None:
        scene_desc = self.data["scenes"][0]["description"] if self.data["scenes"] else "Unknown scene"
        transcript_preview = self.data["audio"]["full_transcript"][:200] if self.data["audio"] else ""
        brief = f"Video shows {scene_desc}."
        if transcript_preview:
            brief += f" Audio content: {transcript_preview}..."
        key_moments = []
        for segment in self.data["audio"].get("segments", [])[:5]:  # First 5 segments
            key_moments.append({
                "timestamp": segment["start"],
                "type": "speech",
                "description": segment["text"][:100]
            })
        track_summary = {}
        for track_id, track in self.data["tracks"].items():
            obj_class = track["class"]
            if obj_class not in track_summary:
                track_summary[obj_class] = 0
            track_summary[obj_class] += 1
        self.data["summary"] = {
            "brief": brief,
            "duration": self.data["video_metadata"].get("duration", 0),
            "scene_count": len(self.data["scenes"]),
            "unique_objects": track_summary,
            "has_audio": bool(self.data["audio"]),
            "key_moments": key_moments
        }
    
    def export_json(self, output_path: str) -> None:
        with open(output_path, 'w') as f:
            json.dump(self.data, f, indent=2)
        print(f"Analysis exported to {output_path}")
    
    def get_data(self) -> Dict:
        return self.data