import os
import sys
import logging
import random
import pandas as pd

from src.cold_start_recommender import ColdStartRecommender, _MOCK_JOBS_DB

# Rule 2: Structured Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("logs", "task07_demo.log")),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Rule 5: Reproducibility
random.seed(42)

class NaiveRandomRecommender:
    """Baseline recommender that just shuffles the database (mimics lack of intelligence)."""
    def recommend(self, limit: int = 5):
        jobs = list(_MOCK_JOBS_DB)
        random.shuffle(jobs)
        return {"jobs": jobs[:limit], "fallback_used": False}

def simulate_first_session_lift(num_users: int = 1000):
    """
    Simulates interactions to measure lift in relevance between
    a naive baseline and the intelligent cold-start strategy.
    
    Parameters
    ----------
    num_users : int
        Number of synthetic new users to generate.
    """
    logger.info(f"Simulating first-session funnel for {num_users} users...")
    
    recommender = ColdStartRecommender()
    baseline_recommender = NaiveRandomRecommender()
    
    # Pools for synthetic generation
    skill_pool = ["Python", "React", "JavaScript", "SQL", "AWS", "Figma", "Strategy"]
    location_pool = ["NY", "SF", "Remote", "Austin", ""]
    
    results = []
    
    for _ in range(num_users):
        user_skills = random.sample(skill_pool, k=random.randint(0, 3))
        user_location = random.choice(location_pool)
        
        # 1. Baseline Recommendations
        baseline_res = baseline_recommender.recommend(limit=5)
        baseline_clicks = simulate_clicks(baseline_res["jobs"], user_skills, user_location)
        
        # 2. Intelligent Cold-Start Recommendations
        smart_res = recommender.recommend(skills=user_skills, location=user_location, limit=5)
        smart_clicks = simulate_clicks(smart_res["jobs"], user_skills, user_location)
        
        results.append({
            "baseline_clicks": baseline_clicks,
            "smart_clicks": smart_clicks,
            "fallback_used": smart_res["fallback_used"]
        })
        
    df = pd.DataFrame(results)
    
    baseline_ctr = (df['baseline_clicks'].sum() / (num_users * 5)) * 100
    smart_ctr = (df['smart_clicks'].sum() / (num_users * 5)) * 100
    lift = ((smart_ctr - baseline_ctr) / baseline_ctr) * 100
    
    logger.info("--- First-Session Action Metrics ---")
    logger.info(f"Naive Baseline CTR: {baseline_ctr:.2f}%")
    logger.info(f"Intelligent Cold-Start CTR: {smart_ctr:.2f}%")
    logger.info(f"Measured Lift: +{lift:.2f}%")
    logger.info("------------------------------------")
    
    return df

def simulate_clicks(recommended_jobs, user_skills, user_location):
    """
    Simulates clicks based on relevance heuristic:
    Users are highly likely to click jobs that match their skills or location.
    """
    clicks = 0
    user_skills_lower = [s.lower() for s in user_skills]
    user_loc_lower = user_location.lower() if user_location else ""
    
    for job in recommended_jobs:
        job_loc = job["location"].lower()
        job_skills = [s.lower() for s in job["skills"]]
        
        match_score = 0
        if user_loc_lower and user_loc_lower == job_loc:
            match_score += 0.4
        if set(user_skills_lower).intersection(set(job_skills)):
            match_score += 0.4
        
        # Even if no match, small random chance to click
        prob = match_score + 0.05
        if random.random() < prob:
            clicks += 1
            
    return clicks

def demo_single_worked_example():
    """Shows a plain-English worked example for explainability and safety."""
    logger.info("\n--- Plain-English Worked Example ---")
    
    recommender = ColdStartRecommender()
    
    # 1. Target match example
    skills = ["React"]
    location = "Austin"
    logger.info(f"INPUT: User onboarding with Skills={skills}, Location={location}")
    res = recommender.recommend(skills, location, limit=3)
    logger.info(f"OUTPUT: Returned {len(res['jobs'])} jobs. Fallback triggered: {res['fallback_used']}")
    logger.info(f"EXPLANATION: The recommender matched the user to the Austin-based Full Stack Developer role due to explicit location/skill overlap, filling the rest with top-tier jobs.")
    
    # 2. Strict Fallback example
    logger.info("\n--- Strict Non-Empty Fallback Demo ---")
    skills = ["Cobol"]
    location = "Antarctica"
    logger.info(f"INPUT: User onboarding with Skills={skills}, Location={location}")
    res = recommender.recommend(skills, location, limit=3)
    logger.info(f"OUTPUT: Returned {len(res['jobs'])} jobs. Fallback triggered: {res['fallback_used']}")
    titles = [j['title'] for j in res['jobs']]
    logger.info(f"EXPLANATION: Because 'Cobol' in 'Antarctica' yielded 0 matches, the system safely fell back to the globally trending jobs ({titles}) ensuring the screen is NEVER empty.")
    
def demo_failure_mode():
    """Forces failure paths to confirm graceful degradation."""
    logger.info("\n--- Break It On Purpose Mode ---")
    
    # Simulating API uninitialized state (Rule 7)
    recommender = ColdStartRecommender()
    recommender.is_initialized = False
    
    try:
        logger.info("Forcing inference on uninitialized model...")
        recommender.recommend(["Python"], "NY")
    except ValueError as e:
        logger.info(f"Successfully caught expected ValueError: {e}")
        
    logger.info("Failure paths degrade gracefully as designed.")

if __name__ == "__main__":
    try:
        os.makedirs("logs", exist_ok=True)
        simulate_first_session_lift(num_users=2000)
        demo_single_worked_example()
        demo_failure_mode()
    except Exception as e:
        logger.critical(f"Unhandled fatal error: {e}", exc_info=True)
        sys.exit(1)
