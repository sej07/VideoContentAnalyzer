import subprocess
import time
import os
import signal
import sys

def run_api():
    print("Starting FastAPI server")
    api_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    return api_process

def run_streamlit():
    print("Starting Streamlit app")
    streamlit_process = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run",
            "app/streamlit_app.py",
            "--server.port=7860",
            "--server.address=0.0.0.0",
            "--server.headless=true",
            "--browser.serverAddress=0.0.0.0",
            "--browser.gatherUsageStats=false"
        ]
    )
    return streamlit_process

def signal_handler(sig, frame):
    print("\nShutting down")
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    api_proc = run_api()
    time.sleep(5)
    streamlit_proc = run_streamlit()
    
    print("Application is running!")
    print("FastAPI: http://localhost:8000")
    print("Streamlit: http://localhost:7860")
    try:
        streamlit_proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        api_proc.terminate()
        streamlit_proc.terminate()