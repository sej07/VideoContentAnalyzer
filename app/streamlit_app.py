import streamlit as st
import requests
import time
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.timeline_generator import create_interactive_timeline

API_URL = "http://localhost:8000"
st.set_page_config(
    page_title="Video Content Analyzer",
    page_icon="🎬",
    layout="wide"
)

st.title("AI-Powered Video Content Analyzer")
st.markdown("Upload a video to get automated analysis with object tracking, transcription, and scene understanding.")
with st.sidebar:
    st.header("About")
    st.markdown("""
    This system analyzes videos using:
    - YOLO for object detection
    - BoTSORT for object tracking
    - Whisper for speech transcription
    - CLIP for scene understanding
    """)
    st.markdown("**Created by:** Sejal Barshikar")
    st.markdown("**Tech Stack:** Python, PyTorch, FastAPI, Streamlit")

tab1, tab2 = st.tabs(["Upload & Analyze", "View Results"])
with tab1:
    st.header("Upload Video")
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv"],
        help="Supported formats: MP4, AVI, MOV, MKV"
    )
    if uploaded_file is not None:
        st.video(uploaded_file)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Analyze Video", use_container_width=True, type="primary"):
                # Upload video to API
                with st.spinner("Uploading video"):
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(f"{API_URL}/upload", files=files)
                    if response.status_code == 200:
                        job_id = response.json()["job_id"]
                        st.session_state["job_id"] = job_id
                        process_response = requests.post(f"{API_URL}/process/{job_id}")
                        if process_response.status_code == 200:
                            st.success(f"Processing started! Job ID: {job_id}")
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            while True:
                                status_response = requests.get(f"{API_URL}/status/{job_id}")
                                status_data = status_response.json()
                                progress = status_data["progress"]
                                message = status_data["message"]
                                status = status_data["status"]
                                progress_bar.progress(progress)
                                status_text.text(f"Status: {message} ({progress}%)")
                                if status == "completed":
                                    st.success("Analysis complete!")
                                    st.balloons()
                                    time.sleep(1)
                                    break
                                elif status == "failed":
                                    st.error(f"Analysis failed: {status_data.get('error', 'Unknown error')}")
                                    break
                                time.sleep(2) 
                        else:
                            st.error("Failed to start processing")
                    else:
                        st.error("Failed to upload video")
with tab2:
    st.header("Analysis Results")
    if "job_id" in st.session_state:
        job_id = st.session_state["job_id"]
        try:
            results_response = requests.get(f"{API_URL}/results/{job_id}")
            if results_response.status_code == 200:
                results = results_response.json()
                # Summary Section
                st.subheader("Summary")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Duration", f"{results['summary']['duration']:.1f}s")
                with col2:
                    st.metric("Scenes", results['summary']['scene_count'])
                with col3:
                    unique_objects = results['summary']['unique_objects']
                    total_objects = sum(unique_objects.values())
                    st.metric("Objects Tracked", total_objects)
                with col4:
                    has_audio = "Yes" if results['summary']['has_audio'] else "No"
                    st.metric("Audio", has_audio)
                st.write(results['summary']['brief'])
                result_tab1, result_tab2, result_tab3, result_tab4 = st.tabs([
                    "Scenes", "Objects", "Transcript", "Download"
                ])
                with result_tab1:
                    st.subheader("Scene Analysis")
                    for scene in results['scenes']:
                        with st.expander(f"Scene {scene['scene_number']}: {scene['description']}"):
                            st.write(f"Frames: {scene['start_frame']} - {scene['end_frame']}")
                            st.write(f"Time: {scene['start_time']:.2f}s - {scene['end_time']:.2f}s")
                            st.write(f"Confidence: {scene['confidence']:.1%}")
                with result_tab2:
                    st.subheader("Object Tracking")
                    st.write("**Unique Objects Detected:**")
                    for obj_class, count in results['summary']['unique_objects'].items():
                        st.write(f"- {obj_class}: {count} unique instances")
                    st.write("Track Details:")
                    for track_id, track_info in results['tracks'].items():
                        with st.expander(f"Track #{track_id} - {track_info['class']}"):
                            st.write(f"First seen: {track_info['first_appearance']:.2f}s")
                            st.write(f"Last seen: {track_info['last_appearance']:.2f}s")
                            st.write(f"Duration: {track_info['duration']:.2f}s")
                            st.write(f"Frames: {track_info['total_frames']}")
                            st.write(f"Avg Confidence: {track_info['avg_confidence']:.1%}")
                with result_tab3:
                    st.subheader("Audio Transcript")
                    if results['audio']['full_transcript']:
                        st.write(f"Language: {results['audio']['language']}")
                        with st.expander("Full Transcript", expanded=True):
                            st.write(results['audio']['full_transcript'])
                        st.write("Timestamped Segments:")
                        for segment in results['audio']['segments']:
                            st.write(f"[{segment['start']:.2f}s - {segment['end']:.2f}s]: {segment['text']}")
                    else:
                        st.info("No audio detected in this video")
                with result_tab4:
                    st.subheader("Download Results")
                    json_str = json.dumps(results, indent=2)
                    st.download_button(
                        label="Download Complete Analysis (JSON)",
                        data=json_str,
                        file_name=f"analysis_{job_id}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                    if results['audio']['full_transcript']:
                        transcript_text = f"Language: {results['audio']['language']}\n\n"
                        transcript_text += "Full Transcript\n"
                        transcript_text += results['audio']['full_transcript']
                        transcript_text += "Timestamped segments\n"    
                        for segment in results['audio']['segments']:
                            transcript_text += f"[{segment['start']:.2f}s - {segment['end']:.2f}s]: {segment['text']}\n"

                        st.download_button(
                            label="Download Transcript (TXT)",
                            data=transcript_text,
                            file_name=f"transcript_{job_id}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
            else:
                st.warning("No results available yet. Upload and analyze a video first!")    
        except Exception as e:
            st.error(f"Error fetching results: {str(e)}")
    else:
        st.info("Upload and analyze a video in the first tab to see results here!")

result_tab1, result_tab2, result_tab3, result_tab4, result_tab5 = st.tabs([
    "Timeline", "Scenes", "Objects", "Transcript", "Download"
])

with result_tab1:
    st.subheader("Interactive Timeline")
    try:
        fig = create_interactive_timeline(results)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        **How to use:**
        - Hover over bars to see object details
        - Scene colors show different scenes
        - Purple bars represent speech segments
        - Red stars mark key moments
        """)
    except Exception as e:
        st.error(f"Error creating timeline: {str(e)}")