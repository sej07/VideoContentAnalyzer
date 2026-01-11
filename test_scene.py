from src.video_processor import extract_frames
from src.scene_understanding import SceneAnalyzer
video_path = "data/sample/test_video2.mp4"
output_dir = "outputs/test_frames2"
print("Extracting frames")
frame_paths = extract_frames(video_path, output_dir, sample_rate=1.0)
print(f"{len(frame_paths)} frames extracted\n")
analyzer = SceneAnalyzer(model_name="ViT-B/32")
print()
print("Detecting scene changes")
boundaries = analyzer.detect_scene_changes(frame_paths, threshold=30.0)
print(f"Found {len(boundaries)} scene(s)\n")
if len(boundaries) <= 2:
    print("Single continuous scene detected.")
    print("Analyzing key moments throughout the talk\n")
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
        print(f"{label.upper()} (Frame {frame_idx})")
        descriptions = analyzer.describe_image(frame_path, prompt_options=ted_talk_prompts)
        top_desc = list(descriptions.items())[0]
        print(f"Description: {top_desc[0]}")
        print(f"Confidence: {top_desc[1]:.1%}\n")

else:
    print("Multiple scenes detected. Running full analysis\n")
    scenes = analyzer.analyze_scenes(frame_paths, scene_threshold=30.0)
    for scene in scenes:
        print(f"SCENE {scene['scene_number']}")
        print(f"Frames: {scene['start_frame']} to {scene['end_frame']}")
        print(f"Description: {scene['description']}")
        print(f"Confidence: {scene['confidence']:.1%}\n")