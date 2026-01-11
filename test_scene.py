"""
Test scene understanding with CLIP.
"""

from src.video_processor import extract_frames
from src.scene_understanding import SceneAnalyzer

# Use TED talk video
video_path = "data/sample/test_video2.mp4"
output_dir = "outputs/test_frames2"

# Extract frames
print("Extracting frames...")
frame_paths = extract_frames(video_path, output_dir, sample_rate=1.0)
print(f"✓ {len(frame_paths)} frames extracted\n")

# Initialize scene analyzer
analyzer = SceneAnalyzer(model_name="ViT-B/32")
print()

# Detect scene changes (TED talks usually have few scene changes)
print("Detecting scene changes...")
boundaries = analyzer.detect_scene_changes(frame_paths, threshold=30.0)
print(f"Found {len(boundaries)} scene(s)\n")

# If only one scene (typical for TED talk), just describe key moments
if len(boundaries) <= 2:  # One continuous scene
    print("Single continuous scene detected.")
    print("Analyzing key moments throughout the talk...\n")
    
    # Analyze first, middle, and last frame
    key_frames = [
        ("Start", frame_paths[0], 0),
        ("Middle", frame_paths[len(frame_paths)//2], len(frame_paths)//2),
        ("End", frame_paths[-1], len(frame_paths)-1)
    ]
    
    ted_talk_prompts = [
        "a person giving a TED talk presentation on stage",
        "a speaker presenting at a conference",
        "someone giving a lecture or educational talk",
        "a person speaking in front of presentation slides",
        "a professional speaker on stage",
        "someone presenting research or ideas to an audience",
        "a public speaking event with a single speaker",
        "a person talking during a seminar or workshop"
    ]
    
    for label, frame_path, frame_idx in key_frames:
        print(f"{'='*60}")
        print(f"{label.upper()} (Frame {frame_idx})")
        print(f"{'='*60}")
        descriptions = analyzer.describe_image(frame_path, prompt_options=ted_talk_prompts)
        
        top_desc = list(descriptions.items())[0]
        print(f"Description: {top_desc[0]}")
        print(f"Confidence: {top_desc[1]:.1%}\n")

else:
    # Multiple scenes detected - full analysis
    print("Multiple scenes detected. Running full analysis...\n")
    scenes = analyzer.analyze_scenes(frame_paths, scene_threshold=30.0)
    
    for scene in scenes:
        print(f"{'='*60}")
        print(f"SCENE {scene['scene_number']}")
        print(f"{'='*60}")
        print(f"Frames: {scene['start_frame']} to {scene['end_frame']}")
        print(f"Description: {scene['description']}")
        print(f"Confidence: {scene['confidence']:.1%}\n")