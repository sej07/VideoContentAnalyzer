import subprocess
import time
import os
import signal
import sys
import requests

def check_api_ready(max_attempts=30):
    print("CHECKING API STATUS")
    
    for i in range(max_attempts):
        try:
            print(f"Attempt {i+1}/{max_attempts}: Checking http://localhost:8000/")
            response = requests.get("http://localhost:8000/", timeout=2)
            if response.status_code == 200:
                print(f"API IS READY! (took {i+1} seconds)")
                return True
        except requests.exceptions.ConnectionError as e:
            print(f"  Connection refused - API not ready yet")
        except Exception as e:
            print(f"  Error: {e}")
        
        time.sleep(1)
    
    print("✗ API FAILED TO START AFTER 30 SECONDS!")
    return False

def run_api():
    print("STARTING FASTAPI SERVER")
    
    try:
        api_process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "api.main:app", 
             "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        import threading
        def log_api_output():
            print("\n[API OUTPUT START]")
            for line in iter(api_process.stdout.readline, ''):
                if line:
                    print(f"[API] {line.strip()}")
        
        log_thread = threading.Thread(target=log_api_output, daemon=True)
        log_thread.start()
        
        print(f"API process started with PID: {api_process.pid}")
        return api_process
        
    except Exception as e:
        print(f"ERROR STARTING API: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_streamlit():
    print("STARTING STREAMLIT")
    
    streamlit_process = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run",
            "app/streamlit_app.py",
            "--server.port=7860",
            "--server.address=0.0.0.0",
            "--server.headless=true",
            "--browser.serverAddress=0.0.0.0",
            "--browser.gatherUsageStats=false",
            "--server.enableCORS=false"
        ]
    )
    return streamlit_process

def signal_handler(sig, frame):
    print("\nShutting down")
    sys.exit(0)

if __name__ == "__main__":
    print("APPLICATION STARTUP")
    
    # Handle signals
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start API
    api_proc = run_api()
    
    if api_proc is None:
        print("FATAL: Could not start API process")
        sys.exit(1)
    
    # Wait for API
    api_ready = check_api_ready(max_attempts=30)
    
    if not api_ready:
        print("FATAL: API did not become ready")
        print("Terminating API process")
        api_proc.terminate()
        sys.exit(1)
    
    # Start Streamlit
    streamlit_proc = run_streamlit()
    
    print("BOTH SERVICES RUNNING")
    print("FastAPI: http://localhost:8000")
    print("Streamlit: http://localhost:7860")
    
    # Wait
    try:
        streamlit_proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down")
    finally:
        api_proc.terminate()
        streamlit_proc.terminate()