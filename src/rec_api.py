import asyncio
import os
import logging
import json
import time
from datetime import datetime
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from typing import List, Dict, Any

from src.inference_engine import OptimizedInferenceEngine

# Rule 2: Structured Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# Telemetry Logger
telemetry_logger = logging.getLogger("telemetry")
telemetry_logger.setLevel(logging.INFO)
os.makedirs("logs", exist_ok=True)
fh = logging.FileHandler(os.path.join("logs", "api_telemetry.log"))
fh.setFormatter(logging.Formatter("%(message)s"))
telemetry_logger.addHandler(fh)

app = FastAPI(title="PlaceMux Inference API", version="1.0.0")

# Global state
engine = OptimizedInferenceEngine()
request_queue = asyncio.Queue()
USE_BATCHING = os.getenv("USE_BATCHING", "0") == "1"
MODEL_AVAILABLE = os.getenv("MODEL_AVAILABLE", "1") == "1"

BATCH_SIZE = 500
BATCH_TIMEOUT = 0.05  # 50ms

@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting API. Batching enabled: {USE_BATCHING}")
    if USE_BATCHING:
        asyncio.create_task(batch_processor())

async def batch_processor():
    """
    Background task that continuously gathers requests from the queue,
    forms a batch, and processes them using the vectorized engine.
    """
    while True:
        batch = []
        try:
            # Wait for at least one item
            item = await request_queue.get()
            batch.append(item)
            
            # Try to grab more up to BATCH_SIZE
            start_time = asyncio.get_event_loop().time()
            while len(batch) < BATCH_SIZE:
                time_left = BATCH_TIMEOUT - (asyncio.get_event_loop().time() - start_time)
                if time_left <= 0:
                    break
                try:
                    next_item = await asyncio.wait_for(request_queue.get(), timeout=time_left)
                    batch.append(next_item)
                except asyncio.TimeoutError:
                    break
            
            # Process batch
            if MODEL_AVAILABLE:
                records = [item[0] for item in batch]
                df = pd.DataFrame(records)
                try:
                    scores = engine.predict(df)
                    for i, (_, future) in enumerate(batch):
                        if not future.done():
                            future.set_result(float(scores[i]))
                except Exception as e:
                    logger.error(f"Batch inference failed: {e}")
                    for _, future in batch:
                        if not future.done():
                            future.set_exception(e)
            else:
                for _, future in batch:
                    if not future.done():
                        future.set_exception(ValueError("Model is currently unavailable."))
                        
        except Exception as e:
            logger.error(f"Batch processor error: {e}")

def log_telemetry(latency_ms: float, status: int, score: float):
    telemetry = {
        "timestamp": datetime.utcnow().isoformat(),
        "latency_ms": latency_ms,
        "http_status": status,
        "prediction_score": score
    }
    telemetry_logger.info(json.dumps(telemetry))

@app.post("/predict")
async def predict(payload: Dict[str, Any], request: Request, response: Response):
    """
    Predict endpoint for scoring a single record.
    """
    start_time = time.time()
    score = 0.0
    status_code = 200
    
    try:
        # Rule 7: API-level Model Unavailability Guard
        if not MODEL_AVAILABLE:
            raise ValueError("Model is currently unavailable.")
            
        # Rule 7: Empty input guard
        if not payload:
            raise ValueError("Empty payload provided.")
            
        if USE_BATCHING:
            # Dynamic Batching Path
            future = asyncio.get_event_loop().create_future()
            await request_queue.put((payload, future))
            score = await future
        else:
            # Baseline Direct Path
            df = pd.DataFrame([payload])
            scores = engine.predict(df)
            score = float(scores[0])
            
    except ValueError as e:
        logger.warning(f"Prediction ValueError: {e}")
        status_code = 503
        response.status_code = 503
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        status_code = 500
        response.status_code = 500
    finally:
        latency_ms = (time.time() - start_time) * 1000
        log_telemetry(latency_ms, status_code, score)
        
    if status_code != 200:
        return {"detail": "Error occurred"}
    return {"score": score}
