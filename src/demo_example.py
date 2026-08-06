import json
import logging
import sys

# Rule 2: Structured Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def demo_worked_example():
    """
    Demonstrates a single worked example of an intelligence defect,
    and shows what happens when the model is unavailable.
    """
    logger.info("--- DEMO: Worked Example of Intelligence Defect ---")
    input_data = {
        "candidate_id": 4059,
        "job_id": 11200,
        "candidate_location": "New York",
        "job_location": "San Francisco",
        "job_is_remote": False,
        "skill_match_score": 0.95
    }
    
    # Model's output
    offline_prediction = 0.96
    
    # Plain English Reason
    reason = "The offline model predicted a 96% match score because it heavily weighted the skill match (95%). However, it failed to account for the fact that the candidate is in New York and the job is on-site in San Francisco, resulting in 0 actual applications for this pair in live traffic."
    
    logger.info(f"Input: {json.dumps(input_data, indent=2)}")
    logger.info(f"Output: Offline Predicted Match Score = {offline_prediction}")
    logger.info(f"Plain-English Reason for Failure: {reason}")
    
    logger.info("--- DEMO: Model Unavailable Path ---")
    model_service_available = False
    
    # Rule 7: API-level Model Unavailability Guard
    try:
        if not model_service_available:
            raise RuntimeError("Model is currently unavailable. Connection timeout.")
        
        # simulated call
        result = 0.96
    except RuntimeError as e:
        logger.warning(f"Prediction aborted: {e}")
        logger.info("Fallback activated: Returning default score of 0.0 to prevent system crash.")
        result = 0.0
        
    logger.info(f"Final Output when model is down: {result}")

if __name__ == "__main__":
    demo_worked_example()
