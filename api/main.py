from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import Optional, Dict
import os
import json
import uuid
from datetime import datetime
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.video_processor import get_video_info, extract_frames
from src.object_tracking import ObjectTracker
from src.audio_processing import extract_audio, AudioTranscriber
from src.scene_understanding import SceneAnalyzer
from src.data_integration import VideoAnalysisIntegrator

app = FastAPI(title="Video Content Analyzer API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
UPLOAD_DIR = Path("data/uploads")
OUTPUT_DIR = Path("outputs/api_results")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
job_status: Dict[str, dict] = {}

def video_has_audio(video_path: str) -> bool:
    import subprocess
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'a:0',
            '-show_entries', 'stream=codec_type',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout.strip() == 'audio'
    except:
        return False
class JobStatus(BaseModel):
    job_id: str
    status: str  # "pending", "processing", "completed", "failed"
    progress: int  # 0-100
    message: str
    result_path: Optional[str] = None
    error: Optional[str] = None
@app.get("/")
def read_root():
    """Health check endpoint."""
    return {
        "status": "online",
        "message": "Video Content Analyzer API",
        "version": "1.0.0"
    }
@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.content_type.startswith("video/"):
        raise HTTPException(400, "File must be a video")
    job_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    try:
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(500, f"Failed to save file: {str(e)}")
    job_status[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0,
        "message": "Video uploaded successfully",
        "filename": file.filename,
        "upload_time": datetime.now().isoformat(),
        "file_path": str(file_path)
    }
    return {
        "job_id": job_id,
        "message": "Video uploaded successfully. Use /process/{job_id} to start analysis."
    }
def process_video_task(job_id: str):
    try:
        job = job_status[job_id]
        video_path = job["file_path"]
        job["status"] = "processing"
        job["progress"] = 5
        job["message"] = "Starting analysis"
        job_output_dir = OUTPUT_DIR / job_id
        job_output_dir.mkdir(exist_ok=True)
        frames_dir = job_output_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        integrator = VideoAnalysisIntegrator()
        job["progress"] = 10
        job["message"] = "Extracting video metadata"
        video_info = get_video_info(video_path)
        integrator.add_video_metadata(video_info)
        job["progress"] = 20
        job["message"] = "Extracting frames"
        frame_paths = extract_frames(video_path, str(frames_dir), sample_rate=1.0)
        job["progress"] = 35
        job["message"] = "Running object detection and tracking..."
        tracker = ObjectTracker(model_name='yolov8n.pt', confidence_threshold=0.5)
        tracking_results = tracker.track_in_frames(frame_paths)
        integrator.add_frame_detections(frame_paths, tracking_results)
        integrator.compute_tracks_summary()
        job["progress"] = 60
        job["message"] = "Transcribing audio"
        if video_has_audio(video_path):
            job["message"] = "Extracting and transcribing audio..."
            audio_path = job_output_dir / "audio.wav"
            try:
                extract_audio(video_path, str(audio_path))
                transcriber = AudioTranscriber(model_name='base')
                transcript = transcriber.transcribe(str(audio_path))
                integrator.add_audio_transcript(transcript)
                os.remove(audio_path)  # Cleanup
            except Exception as e:
                job["message"] = f"Audio processing failed: {str(e)}, continuing without audio..."
                integrator.add_audio_transcript({
                    'language': 'none',
                    'text': '',
                    'segments': []
                })
        else:
            job["message"] = "No audio stream detected, skipping transcription..."
            integrator.add_audio_transcript({
                'language': 'none',
                'text': '',
                'segments': []
            })
        job["progress"] = 80
        job["message"] = "Analyzing scenes with CLIP"
        analyzer = SceneAnalyzer(model_name="ViT-B/32")
        scenes = analyzer.analyze_scenes(frame_paths, scene_threshold=30.0)
        integrator.add_scenes(scenes)
        job["progress"] = 90
        job["message"] = "Generating summary"
        integrator.generate_summary()
        job["progress"] = 95
        job["message"] = "Exporting results"
        result_path = job_output_dir / "analysis_results.json"
        integrator.export_json(str(result_path))
        job["status"] = "completed"
        job["progress"] = 100
        job["message"] = "Analysis complete!"
        job["result_path"] = str(result_path)
        job["completion_time"] = datetime.now().isoformat()
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        job["message"] = f"Analysis failed: {str(e)}"
@app.post("/process/{job_id}")
async def process_video(job_id: str, background_tasks: BackgroundTasks):
    if job_id not in job_status:
        raise HTTPException(404, "Job ID not found")
    job = job_status[job_id]
    if job["status"] != "pending":
        raise HTTPException(400, f"Job already {job['status']}")
    background_tasks.add_task(process_video_task, job_id)
    return {
        "job_id": job_id,
        "message": "Processing started. Use /status/{job_id} to check progress."
    }
@app.get("/status/{job_id}")
def get_status(job_id: str):
    if job_id not in job_status:
        raise HTTPException(404, "Job ID not found")
    job = job_status[job_id]
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "progress": job["progress"],
        "message": job["message"],
        "error": job.get("error")
    }
@app.get("/results/{job_id}")
def get_results(job_id: str):
    if job_id not in job_status:
        raise HTTPException(404, "Job ID not found")
    job = job_status[job_id]
    if job["status"] != "completed":
        raise HTTPException(400, f"Job is {job['status']}, not completed")
    result_path = job.get("result_path")
    if not result_path or not os.path.exists(result_path):
        raise HTTPException(404, "Results file not found")
    with open(result_path, 'r') as f:
        results = json.load(f)
    return results
@app.get("/download/{job_id}")
def download_results(job_id: str):
    if job_id not in job_status:
        raise HTTPException(404, "Job ID not found")
    job = job_status[job_id]
    if job["status"] != "completed":
        raise HTTPException(400, f"Job is {job['status']}, not completed")
    result_path = job.get("result_path")
    if not result_path or not os.path.exists(result_path):
        raise HTTPException(404, "Results file not found")
    return FileResponse(
        result_path,
        media_type="application/json",
        filename=f"analysis_{job_id}.json"
    )

if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000)