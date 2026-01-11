import clip 
import torch
import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple
from pathlib import Path

class SceneAnalyzer:
    def __init__(self, model_name:str = 'ViT-B/32'):
        print(f"Loading CLIP model: {model_name}")
        self.device = 'mps' if torch.backends.mps.is_available else 'cpu'
        self.model, self.preprocess = clip.load(model_name, device = self.device)
        print(f"CLIP model loaded on {self.device}")

    def detect_scene_changes(self, frame_paths: List[str], threshold: float = 30.0) -> List[int]:
        print(f"Detecting scene changesin {len(frame_paths)} frames")
        scene_boundaries = [0]
        prev_frame = None
        for i, frame_path in enumerate(frame_paths):
            frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            if prev_frame is not None:
                diff = cv2.absdiff(prev_frame, frame)
                mean_diff = np.mean(diff)
                if mean_diff > threshold: 
                    scene_boundaries.append()
                    print(f"Scene change detected at frame {i} (diff: {mean_diff:.2f})")
                prev_frame = frame
        print(f"Found {len(scene_boundaries)} scenes")
        return scene_boundaries
    
    def describe_image(self, image_path:str, prompt_options: List[str] = None)-> Dict[str, float]:
        if prompt_options is None: 
            prompt_options = [
            # People activities
            "people talking indoors",
            "people talking outdoors", 
            "people working in an office",
            "people exercising or playing sports",
            "people eating or dining",
            "people performing or presenting on stage",
            "people shopping or in a store",
            "a person speaking or giving a presentation",
            
            # Settings
            "an indoor scene with people",
            "an outdoor scene with people",
            "a crowded public space",
            "a quiet indoor environment",
            "an urban street scene",
            "a natural outdoor environment",
            "a professional workplace setting",
            "a recreational or leisure activity",
            
            # Events
            "a social gathering or party",
            "a business meeting or conference",
            "a sports event or game",
            "a performance or concert",
            "a classroom or educational setting",
            "a transportation scene with vehicles",
            
            # Nature & Objects
            "a landscape or nature scene",
            "animals in their natural habitat",
            "vehicles on a road or street",
            "architecture or buildings",
            "food or cooking",
            "technology or electronics",
            
            # Activities
            "someone creating or making something",
            "people traveling or commuting",
            "a celebration or special event",
            "daily life activities"
        ]
        image = Image.open(image_path)
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        text_tokens = clip.tokenize(prompt_options).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_tokens)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        results = {}
        for i, prompt in enumerate(prompt_options):
            results[prompt] = float(similarity[0, i])
        results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
        return results
    
    def analyze_scenes(self, frame_paths: List[str], scene_threshold: float = 30.0) -> List[Dict]:
        boundaries = self.detect_scene_changes(frame_paths, scene_threshold)
        scenes = []
        print(f"\nAnalyzing {len(boundaries)} scenes with CLIP")
        for i, start_idx in enumerate(boundaries):
            if i + 1 < len(boundaries):
                end_idx = boundaries[i + 1] - 1
            else:
                end_idx = len(frame_paths) - 1
            key_frame_idx = (start_idx + end_idx) // 2
            key_frame_path = frame_paths[key_frame_idx]
            descriptions = self.describe_image(key_frame_path)
            best_description = list(descriptions.keys())[0]
            confidence = descriptions[best_description]
            scene = {
                'scene_number': i + 1,
                'start_frame': start_idx,
                'end_frame': end_idx,
                'key_frame_index': key_frame_idx,
                'key_frame_path': key_frame_path,
                'description': best_description,
                'confidence': confidence,
                'all_descriptions': descriptions
            }
            scenes.append(scene)
            print(f"Scene {i+1}: Frames {start_idx}-{end_idx}")
            print(f"  Description: {best_description}")
            print(f"  Confidence: {confidence:.1%}")
            print()
        print("Scene analysis complete!")
        return scenes
