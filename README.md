---
title: Video Content Analyzer
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# AI-Powered Video Content Analyzer

An advanced video analysis system that combines multiple AI models to automatically analyze videos with object detection, tracking, speech transcription, and scene understanding.

## Features

- **Object Detection & Tracking**: Real-time object detection using YOLOv8 with persistent tracking across frames
- **Speech Transcription**: Automatic audio transcription using OpenAI Whisper with word-level timestamps
- **Scene Understanding**: Natural language scene descriptions using CLIP
- **Interactive Timeline**: Multi-track visualization showing objects, scenes, and transcript segments
- **REST API**: FastAPI backend for programmatic access

## Technologies

- **Computer Vision**: YOLOv8, BoTSORT
- **NLP & Audio**: Whisper, CLIP
- **Backend**: FastAPI, Python
- **Frontend**: Streamlit, Plotly
- **Deployment**: Docker, Hugging Face Spaces

## How It Works

1. Upload a video through the web interface
2. System extracts frames and processes video in parallel:
   - Object detection and tracking
   - Audio extraction and transcription
   - Scene analysis and description
3. Results are integrated into a unified timeline
4. Interactive visualization with synchronized playback

## Use Cases

- Video indexing and search
- Content moderation
- Accessibility (automatic subtitles)
- Sports analytics
- Security footage analysis
- Educational content analysis

## Author

**Sejal Barshikar**
- MS in Computer Science @ Northeastern University
- Specialization