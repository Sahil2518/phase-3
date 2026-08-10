import logging
import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from src.cold_start_recommender import ColdStartRecommender

# Rule 2: Structured Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("logs", "task07_api.log")),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="PlaceMux Cold-Start API", version="1.0.0")

# Global instances
recommender = ColdStartRecommender()
MODEL_AVAILABLE = os.getenv("MODEL_AVAILABLE", "1") == "1"

class OnboardingPayload(BaseModel):
    skills: List[str] = []
    location: Optional[str] = ""

@app.post("/recommend/cold-start")
def get_cold_start_recommendations(payload: OnboardingPayload):
    """
    Endpoint for fetching first-session job recommendations for a brand new user.
    """
    # Rule 7: API-level Model Unavailability Guard
    if not MODEL_AVAILABLE:
        logger.error("API call failed: Model unavailable.")
        raise HTTPException(status_code=503, detail="Model is currently unavailable.")
        
    try:
        result = recommender.recommend(
            skills=payload.skills,
            location=payload.location,
            limit=5
        )
        return {
            "status": "success",
            "fallback_used": result["fallback_used"],
            "recommended_jobs": result["jobs"]
        }
    except ValueError as e:
        logger.warning(f"Prediction ValueError: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        # Catch unexpected errors to prevent API crashes
        logger.error(f"Unexpected error in /recommend/cold-start: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
