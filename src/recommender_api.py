import os
import time
import logging
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, Request, Response

from src.two_sided_recommender import CatalogManager, TwoSidedRecommender, BaselineRecommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="PlaceMux Recommendation Engine API", version="1.0.0")

# Global state for fast serving
catalog = None
engine = None
baseline = None

@app.on_event("startup")
async def startup_event():
    global catalog, engine, baseline
    logger.info("Starting Recommendation API...")
    catalog = CatalogManager(random_state=42)
    catalog.generate_catalogs(num_candidates=5000, num_jobs=2000)
    engine = TwoSidedRecommender(catalog)
    baseline = BaselineRecommender(catalog)
    logger.info("Recommendation Engine Ready.")

@app.get("/recommend/jobs")
async def recommend_jobs(candidate_id: str, k: int = 10):
    start_time = time.time()
    try:
        MODEL_AVAILABLE = os.getenv("MODEL_AVAILABLE", "1") == "1"
        if not MODEL_AVAILABLE:
            # Fallback path
            logger.warning("Model unavailable, using baseline.")
            recs = baseline.recommend_jobs_for_candidate(candidate_id, k)
            source = "baseline"
        else:
            recs = engine.recommend_jobs_for_candidate(candidate_id, k)
            source = "personalized"
            
        latency_ms = (time.time() - start_time) * 1000
        return {
            "candidate_id": candidate_id,
            "latency_ms": latency_ms,
            "source": source,
            "recommendations": recs
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error serving jobs for {candidate_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/recommend/candidates")
async def recommend_candidates(job_id: str, k: int = 10):
    start_time = time.time()
    try:
        MODEL_AVAILABLE = os.getenv("MODEL_AVAILABLE", "1") == "1"
        if not MODEL_AVAILABLE:
            # Fallback path
            logger.warning("Model unavailable, using baseline.")
            recs = baseline.recommend_candidates_for_job(job_id, k)
            source = "baseline"
        else:
            recs = engine.recommend_candidates_for_job(job_id, k)
            source = "personalized"
            
        latency_ms = (time.time() - start_time) * 1000
        return {
            "job_id": job_id,
            "latency_ms": latency_ms,
            "source": source,
            "recommendations": recs
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error serving candidates for {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
