import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime, timedelta

# Rule 2: Structured Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def generate_interaction_logs(num_records=10000, random_state=42):
    """
    Generates synthetic interaction logs simulating live traffic in Phase 3.
    Includes intentional defects where offline score overestimates online performance.

    Parameters
    ----------
    num_records : int
        Number of interaction records to generate.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    df : pd.DataFrame
        Dataframe containing the generated logs.
    """
    np.random.seed(random_state)
    logger.info(f"Generating {num_records} synthetic interaction logs...")

    locations = ['New York', 'San Francisco', 'Austin', 'Remote', 'London']
    seniorities = ['Entry Level', 'Mid Level', 'Senior', 'Director']
    
    data = {
        'interaction_id': [f"evt_{i:07d}" for i in range(num_records)],
        'candidate_id': np.random.randint(1000, 5000, num_records),
        'job_id': np.random.randint(10000, 15000, num_records),
        'candidate_location': np.random.choice(locations, num_records),
        'job_location': np.random.choice(locations, num_records),
        'candidate_seniority': np.random.choice(seniorities, num_records),
        'job_seniority': np.random.choice(seniorities, num_records),
    }
    
    df = pd.DataFrame(data)
    
    # Calculate an underlying "true" match score (0 to 1) based on pure skill match (randomized here)
    df['skill_match_score'] = np.random.beta(a=2, b=5, size=num_records)
    
    # The Offline Model heavily relies on skill match, but ignores location and seniority mismatch.
    # We add some noise to make it realistic.
    df['offline_score'] = df['skill_match_score'] + np.random.normal(0, 0.05, num_records)
    df['offline_score'] = df['offline_score'].clip(0.0, 1.0)
    
    # Let's define the "Online conversion" (is_apply).
    # Actual conversion heavily depends on location and seniority matching, which the offline model missed.
    
    # Feature 1: Location Match (Remote jobs match anyone)
    df['is_location_match'] = (df['candidate_location'] == df['job_location']) | (df['job_location'] == 'Remote')
    
    # Feature 2: Seniority Match
    seniority_map = {'Entry Level': 1, 'Mid Level': 2, 'Senior': 3, 'Director': 4}
    df['cand_sen_level'] = df['candidate_seniority'].map(seniority_map)
    df['job_sen_level'] = df['job_seniority'].map(seniority_map)
    
    # Candidate applying to a role too senior for them
    df['is_underqualified'] = df['cand_sen_level'] < df['job_sen_level']
    # Candidate applying to a role too junior (they might not want it)
    df['is_overqualified'] = df['cand_sen_level'] > df['job_sen_level']
    
    # True online conversion probability
    # Base probability driven by skill
    true_prob = df['skill_match_score'].copy()
    
    # Penalty for location mismatch
    true_prob = np.where(~df['is_location_match'], true_prob * 0.05, true_prob)
    
    # Penalty for being underqualified
    true_prob = np.where(df['is_underqualified'], true_prob * 0.1, true_prob)
    
    # Penalty for being overqualified
    true_prob = np.where(df['is_overqualified'], true_prob * 0.4, true_prob)
    
    df['true_conversion_prob'] = true_prob.clip(0.0, 1.0)
    
    # Simulate clicks and applies
    # Apply requires a click first.
    click_prob = np.clip(df['true_conversion_prob'] * 2.5, 0.0, 1.0) # Higher than apply
    df['is_click'] = np.random.binomial(1, click_prob)
    
    # Apply only if clicked
    apply_prob = np.where(df['is_click'] == 1, df['true_conversion_prob'] / (click_prob + 1e-9), 0)
    df['is_apply'] = np.random.binomial(1, np.clip(apply_prob, 0.0, 1.0))
    
    # Add timestamps (last 7 days)
    base_time = datetime.now() - timedelta(days=7)
    df['timestamp'] = [base_time + timedelta(minutes=int(m)) for m in np.random.randint(0, 7*24*60, num_records)]
    
    # Drop temporary columns used for generation
    df = df.drop(columns=['skill_match_score', 'cand_sen_level', 'job_sen_level', 'true_conversion_prob'])
    
    logger.info("Simulation complete.")
    return df

def main():
    try:
        os.makedirs("logs", exist_ok=True)
        df = generate_interaction_logs()
        
        # Rule 2: Data Validation Guards
        assert df.shape[0] > 0, "Generated dataset is empty!"
        assert not df.isnull().all(axis=1).any(), "Rows with all-NaN values found."
        
        out_path = "logs/interaction_logs.csv"
        df.to_csv(out_path, index=False)
        logger.info(f"Saved simulated logs to {out_path}")
        
    except Exception as e:
        logger.critical(f"Unhandled fatal error in data simulation: {e}", exc_info=True)
        import sys
        sys.exit(1)

if __name__ == "__main__":
    main()
