import os
import subprocess
import time
import requests
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("logs", "task05.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

PORT = 8000
BASE_URL = f"http://127.0.0.1:{PORT}"

def start_server(use_batching: bool, model_available: bool = True) -> subprocess.Popen:
    env = os.environ.copy()
    env["USE_BATCHING"] = "1" if use_batching else "0"
    env["MODEL_AVAILABLE"] = "1" if model_available else "0"
    
    cmd = [
        sys.executable, "-m", "uvicorn", 
        "src.rec_api:app", "--host", "127.0.0.1", "--port", str(PORT)
    ]
    process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    started = False
    for _ in range(30):
        try:
            resp = requests.get(f"{BASE_URL}/docs")
            if resp.status_code == 200:
                started = True
                break
        except requests.ConnectionError:
            pass
        time.sleep(0.5)
        
    if not started:
        logger.error("Failed to start FastAPI server.")
        process.kill()
        sys.exit(1)
        
    return process

def run_load_test(run_time: str = "15s", users: int = 250, spawn_rate: int = 50):
    logger.info(f"Running load test: {users} users, {spawn_rate} spawn rate, {run_time} duration")
    cmd = [
        sys.executable, "-m", "locust", 
        "-f", "src/locustfile.py", 
        "--headless", 
        "-u", str(users), 
        "-r", str(spawn_rate), 
        "--run-time", run_time,
        "--host", BASE_URL,
        "--only-summary"
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Locust load test failed: {e.stderr}")

def main():
    os.makedirs("logs", exist_ok=True)
    telemetry_path = os.path.join("logs", "api_telemetry.log")
    if os.path.exists(telemetry_path):
        os.remove(telemetry_path)
    
    logger.info("===============================================")
    logger.info("Phase 3, Task 5: Capstone Reliability Sign-off")
    logger.info("===============================================")
    
    # 1. Start Integrated API
    logger.info("\n--- Phase A: Integrated Scale Test (Dynamic Batching + Telemetry) ---")
    server = start_server(use_batching=True)
    logger.info("Server started. Emitting telemetry to api_telemetry.log")
    
    try:
        run_load_test()
    finally:
        server.terminate()
        server.wait()
        
    # 2. Failure Injection Test
    logger.info("\n--- Phase B: Failure Injection Test (Model Unavailable) ---")
    server = start_server(use_batching=True, model_available=False)
    try:
        # Give it a second to stabilize
        time.sleep(1)
        resp = requests.post(f"{BASE_URL}/predict", json={"skill_score": 0.5, "experience_years": 5})
        logger.info(f"Fallback HTTP Status: {resp.status_code}")
        if resp.status_code == 503:
            logger.info("Graceful degradation confirmed.")
        else:
            logger.error("Expected HTTP 503.")
    except Exception as e:
        logger.error(f"Failure injection failed: {e}")
    finally:
        server.terminate()
        server.wait()
        
    # 3. Generate Sign-off Report
    logger.info("\n--- Phase C: Generating Reliability Sign-off ---")
    cmd = [sys.executable, "-m", "src.generate_signoff"]
    subprocess.run(cmd, check=True)
    
    logger.info("\n===============================================")
    logger.info("Task 5 Capstone Complete.")
    logger.info("===============================================")

if __name__ == "__main__":
    main()
