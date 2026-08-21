import os
import sys
import time
import logging
from typing import Dict, Any, List
from fastapi import FastAPI, Request, HTTPException, Depends, Header
from pydantic import BaseModel, Field

# Rule 2: Structured Logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("logs", "task17.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Partner API Metadata for OpenAPI documentation
tags_metadata = [
    {
        "name": "Matching",
        "description": "Core intelligence endpoints for candidate-to-job matching and scoring.",
    },
]

app = FastAPI(
    title="PlaceMux ATS Partner API",
    description="Public API surface for ATS integrations, providing versioned scoring, matching, and explainability.",
    version="1.0.0",
    openapi_tags=tags_metadata,
)

# ---------------------------------------------------------------------------
# Rate Limiting & Quota Enforcement (In-Memory)
# ---------------------------------------------------------------------------
class PartnerRateLimiter:
    """
    In-memory rate limiter to prevent abuse and model scraping.
    Uses a simple sliding window per API key.
    """
    def __init__(self, limit: int = 5, window_seconds: int = 60):
        self.limit = limit
        self.window_seconds = window_seconds
        # dict mapping API key -> list of timestamps
        self.requests: Dict[str, List[float]] = {}
        
    def check_rate_limit(self, api_key: str) -> None:
        """
        Check if the API key has exceeded its rate limit.
        
        Parameters
        ----------
        api_key : str
            The caller's API key.
            
        Raises
        ------
        HTTPException
            If the API key is missing or the rate limit is exceeded (429).
        """
        if not api_key:
            raise HTTPException(status_code=401, detail="Missing x-api-key header")
            
        now = time.time()
        if api_key not in self.requests:
            self.requests[api_key] = []
            
        # Clean up old requests outside the window
        self.requests[api_key] = [t for t in self.requests[api_key] if now - t < self.window_seconds]
        
        if len(self.requests[api_key]) >= self.limit:
            logger.warning(f"Rate limit exceeded for API Key: {api_key}")
            raise HTTPException(
                status_code=429, 
                detail=f"Rate limit exceeded. Quota is {self.limit} requests per {self.window_seconds} seconds."
            )
            
        self.requests[api_key].append(now)

# Global rate limiter instance: 5 requests per 10 seconds for demo purposes
rate_limiter = PartnerRateLimiter(limit=5, window_seconds=10)

def verify_api_key(x_api_key: str = Header(...)):
    """Dependency to verify API key and enforce rate limits."""
    # In a real system, this would check against a DB of valid partner keys.
    valid_keys = {"partner_a_live", "partner_b_live"}
    if x_api_key not in valid_keys:
        raise HTTPException(status_code=403, detail="Invalid API Key")
        
    rate_limiter.check_rate_limit(x_api_key)
    return x_api_key

# ---------------------------------------------------------------------------
# API Models
# ---------------------------------------------------------------------------
class MatchRequest(BaseModel):
    candidate_skills: List[str] = Field(..., description="List of candidate skills")
    job_skills: List[str] = Field(..., description="List of required job skills")
    candidate_seniority: int = Field(..., description="Candidate seniority level (1-4)")
    job_seniority: int = Field(..., description="Required job seniority level (1-4)")

class MatchResponse(BaseModel):
    score: float = Field(..., description="Matching score between 0.0 and 1.0")
    explanation: str = Field(..., description="Plain-English reason for the score")
    partner_id: str = Field(..., description="The ID of the calling partner")

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/api/v1/match", response_model=MatchResponse, tags=["Matching"])
async def compute_match(payload: MatchRequest, api_key: str = Depends(verify_api_key)):
    """
    Computes a match score and explanation between a candidate and a job.
    Version 1 of the public scoring endpoint for ATS partners.
    """
    # Rule 7: Empty input guard
    if not payload.candidate_skills or not payload.job_skills:
        logger.warning("Empty skills provided in request.")
        return MatchResponse(score=0.0, explanation="Insufficient data for matching.", partner_id=api_key)
        
    try:
        # Simplified scoring logic representing the Intelligence Layer
        skill_overlap = len(set(payload.candidate_skills).intersection(set(payload.job_skills)))
        skill_score = skill_overlap / max(len(payload.job_skills), 1)
        
        sen_diff = payload.candidate_seniority - payload.job_seniority
        if sen_diff < 0:
            sen_score = 0.2
        elif sen_diff > 0:
            sen_score = 0.5
        else:
            sen_score = 1.0
            
        final_score = (0.7 * skill_score) + (0.3 * sen_score)
        
        # Rule 7: Bounds checking
        final_score = max(0.0, min(1.0, final_score))
        
        reasons = []
        if sen_score == 1.0:
            reasons.append("Exact seniority match.")
        if skill_score > 0.5:
            reasons.append(f"Strong skill overlap ({int(skill_score*100)}%).")
            
        explanation = "Good match because: " + " ".join(reasons) if reasons else "Partial match based on profile."
        
        return MatchResponse(score=final_score, explanation=explanation, partner_id=api_key)
        
    except Exception as e:
        logger.error(f"Internal prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error during inference.")
