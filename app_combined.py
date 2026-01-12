import subprocess
import time
import threading
import sys

def run_api():
    subprocess.run([sys.executable, "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"])

def run_streamlit():
    time.sleep(3)  
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0"])

if __name__ == "__main__":
    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()
    
    run_streamlit()