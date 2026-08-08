import os
import subprocess
import time
import requests
import json
import logging
import sys

# Rule 2: Structured Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("logs", "task04.log")),
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
    
    # We use Uvicorn directly via module
    cmd = [
        sys.executable, "-m", "uvicorn", 
        "src.rec_api:app", "--host", "127.0.0.1", "--port", str(PORT)
    ]
    process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for server to start
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

def run_load_test(run_time: str = "10s", users: int = 200, spawn_rate: int = 50) -> float:
    """
    Runs locust load test and returns the 95th percentile latency (ms).
    """
    logger.info(f"Running load test: {users} users, {spawn_rate} spawn rate, {run_time} duration")
    cmd = [
        sys.executable, "-m", "locust", 
        "-f", "src/locustfile.py", 
        "--headless", 
        "-u", str(users), 
        "-r", str(spawn_rate), 
        "--run-time", run_time,
        "--host", BASE_URL,
        "--csv", "logs/locust_stats",
        "--only-summary"
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Locust load test failed: {e.stderr}")
        return float('inf')
        
    # Read the summary CSV to get p95 latency
    try:
        with open("logs/locust_stats_stats.csv", "r") as f:
            lines = f.readlines()
            if not lines:
                return float('inf')
            
            header = lines[0].strip().split(',')
            try:
                p95_index = header.index("95%")
            except ValueError:
                p95_index = 16 # Fallback
                
            for line in lines[1:]:
                if 'POST' in line and '/predict' in line:
                    parts = line.strip().split(',')
                    if len(parts) > p95_index:
                        try:
                            return float(parts[p95_index])
                        except ValueError:
                            pass
    except Exception as e:
        logger.error(f"Could not read locust CSV: {e}")
        
    return float('inf')

def main():
    os.makedirs("logs", exist_ok=True)
    
    logger.info("===============================================")
    logger.info("Phase 3, Task 4: Load Testing & Scaling Plan")
    logger.info("===============================================")
    
    # 1. Baseline Test (No Batching)
    logger.info("\n--- Phase A: Baseline Inference (Row-by-Row REST) ---")
    server = start_server(use_batching=False)
    logger.info("Server started in DIRECT mode.")
    try:
        p95_baseline = run_load_test(run_time="10s", users=250, spawn_rate=50)
        logger.info(f"Baseline Load Test Complete. p95 Latency: {p95_baseline} ms")
    finally:
        server.terminate()
        server.wait()
        
    # 2. Scaling Plan (Dynamic Batching)
    logger.info("\n--- Phase B: Scaling Plan (Dynamic Batching) ---")
    server = start_server(use_batching=True)
    logger.info("Server started in DYNAMIC BATCHING mode.")
    try:
        p95_batched = run_load_test(run_time="10s", users=250, spawn_rate=50)
        logger.info(f"Batched Load Test Complete. p95 Latency: {p95_batched} ms")
        
        improvement = ((p95_baseline - p95_batched) / p95_baseline) * 100 if p95_baseline > 0 else 0
        logger.info(f"Latency Improvement: {improvement:.2f}% reduction at {250} concurrent users")
        
    finally:
        server.terminate()
        server.wait()
        
    # 3. Failure Path Test
    logger.info("\n--- Phase C: Failure Path (Model Unavailable) ---")
    server = start_server(use_batching=True, model_available=False)
    logger.info("Server started with MODEL UNAVAILABLE.")
    try:
        resp = requests.post(f"{BASE_URL}/predict", json={"skill_score": 0.5, "experience_years": 5})
        logger.info(f"Failure Path Response Code: {resp.status_code}")
        logger.info(f"Failure Path Response Body: {resp.text}")
        if resp.status_code == 503:
            logger.info("Graceful degradation verified.")
        else:
            logger.error("Expected HTTP 503 but got something else.")
    except Exception as e:
        logger.error(f"Failure path test error: {e}")
    finally:
        server.terminate()
        server.wait()
        
    logger.info("\n===============================================")
    logger.info("Task 4 Execution Complete. Review logs/ for details.")
    logger.info("===============================================")

if __name__ == "__main__":
    main()
