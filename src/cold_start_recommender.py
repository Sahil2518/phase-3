import logging
import random
import uuid

# Rule 2: Structured Logging
logger = logging.getLogger(__name__)

# Mock database of jobs for the cold-start recommender
_MOCK_JOBS_DB = [
    {"job_id": "j_1001", "title": "Frontend Developer", "location": "NY", "skills": ["React", "JavaScript", "CSS"], "is_trending": True, "score": 0.9},
    {"job_id": "j_1002", "title": "Backend Engineer", "location": "SF", "skills": ["Python", "Django", "SQL"], "is_trending": True, "score": 0.88},
    {"job_id": "j_1003", "title": "Data Scientist", "location": "NY", "skills": ["Python", "Pandas", "Machine Learning"], "is_trending": False, "score": 0.85},
    {"job_id": "j_1004", "title": "Product Manager", "location": "Remote", "skills": ["Agile", "Scrum", "Strategy"], "is_trending": True, "score": 0.92},
    {"job_id": "j_1005", "title": "DevOps Engineer", "location": "SF", "skills": ["AWS", "Docker", "Kubernetes"], "is_trending": False, "score": 0.84},
    {"job_id": "j_1006", "title": "Full Stack Developer", "location": "Austin", "skills": ["React", "Node.js", "MongoDB"], "is_trending": True, "score": 0.91},
    {"job_id": "j_1007", "title": "UX Designer", "location": "NY", "skills": ["Figma", "UI/UX", "Wireframing"], "is_trending": False, "score": 0.82},
    {"job_id": "j_1008", "title": "Machine Learning Engineer", "location": "Remote", "skills": ["Python", "TensorFlow", "PyTorch"], "is_trending": True, "score": 0.95}
]

class ColdStartRecommender:
    """
    Provides initial job recommendations for brand new candidates (cold-start).
    
    Uses onboarding inputs (skills, location) to filter a pool of high-quality
    jobs. Implements a strict, non-empty fallback to globally trending jobs 
    if the targeted search fails or yields no results.
    """
    
    def __init__(self):
        """Initializes the recommender with pre-loaded trending jobs."""
        self.is_initialized = True
        self.trending_jobs = sorted(
            [j for j in _MOCK_JOBS_DB if j["is_trending"]],
            key=lambda x: x["score"], 
            reverse=True
        )

    def recommend(self, skills: list, location: str, limit: int = 5) -> dict:
        """
        Recommends jobs based on onboarding data, with a guaranteed fallback.
        
        Parameters
        ----------
        skills : list of str
            The skills extracted during onboarding.
        location : str
            The candidate's preferred location.
        limit : int
            Maximum number of jobs to return.
            
        Returns
        -------
        dict
            Contains 'jobs' (list) and 'fallback_used' (bool).
        """
        # Rule 7: None / Uninitialized Model Guard
        if not self.is_initialized:
            raise ValueError("Cannot predict: model is uninitialized or None.")
            
        # Standardize inputs
        skills = [s.lower() for s in skills] if skills else []
        location = location.lower() if location else ""
        
        # Rule 7: Empty Input Guard
        if not skills and not location:
            logger.warning("Empty onboarding input provided. Defaulting directly to fallback.")
            return {"jobs": self.trending_jobs[:limit], "fallback_used": True}
            
        try:
            # Attempt targeted match
            matched_jobs = []
            for job in _MOCK_JOBS_DB:
                job_loc = job["location"].lower()
                job_skills = [s.lower() for s in job["skills"]]
                
                # Match logic: exact location match OR overlapping skills
                loc_match = (location == job_loc) if location else False
                skill_match = bool(set(skills).intersection(set(job_skills)))
                
                # Boost score slightly if both match
                if loc_match or skill_match:
                    job_copy = dict(job)
                    if loc_match and skill_match:
                        job_copy["score"] = min(1.0, job_copy["score"] + 0.05)
                    matched_jobs.append(job_copy)
            
            # Sort by score
            matched_jobs = sorted(matched_jobs, key=lambda x: x["score"], reverse=True)
            
            # Fallback if no targeted matches found
            if not matched_jobs:
                logger.info(f"No targeted matches found for loc='{location}', skills={skills}. Triggering fallback.")
                return {"jobs": self.trending_jobs[:limit], "fallback_used": True}
                
            # Pad with trending jobs if targeted matches are fewer than limit
            if len(matched_jobs) < limit:
                needed = limit - len(matched_jobs)
                existing_ids = {j["job_id"] for j in matched_jobs}
                padding = [j for j in self.trending_jobs if j["job_id"] not in existing_ids]
                matched_jobs.extend(padding[:needed])
                
            return {"jobs": matched_jobs[:limit], "fallback_used": False}
            
        except Exception as e:
            # Rule 2: Non-fatal step guard - gracefully catch any complex scoring error and return fallback
            logger.error(f"Error during targeted matching: {e}. Defaulting to fallback.", exc_info=True)
            return {"jobs": self.trending_jobs[:limit], "fallback_used": True}
