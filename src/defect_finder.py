import pandas as pd
import json
import logging
import os
import sys

# Rule 2: Structured Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/task01.log", mode='a') if os.path.exists("logs") else logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def identify_defects(df: pd.DataFrame, high_score_threshold=0.7):
    """
    Identifies intelligence defects by finding cohorts with high offline 
    scores but low actual conversion rates.

    Parameters
    ----------
    df : pd.DataFrame
        Interaction logs.
    high_score_threshold : float
        Threshold above which a prediction is considered "confident".

    Returns
    -------
    defects : list
        A list of dictionaries representing ranked defects.
    """
    # Filter for high-confidence predictions
    high_conf_df = df[df['offline_score'] >= high_score_threshold].copy()
    
    if high_conf_df.empty:
        logger.warning("No high confidence predictions found. Cannot analyze defects.")
        return []

    overall_apply_rate = high_conf_df['is_apply'].mean()
    defects = []
    
    # Defect Hypothesis 1: Location Mismatch (excluding Remote)
    loc_mismatch_mask = (high_conf_df['candidate_location'] != high_conf_df['job_location']) & (high_conf_df['job_location'] != 'Remote')
    loc_mismatch_df = high_conf_df[loc_mismatch_mask]
    if len(loc_mismatch_df) > 0:
        apply_rate = loc_mismatch_df['is_apply'].mean()
        defects.append({
            "defect_id": "D-001",
            "name": "Location Mismatch Ignored",
            "description": "Model gives high scores to candidates for on-site roles in different cities.",
            "impacted_volume": int(len(loc_mismatch_df)),
            "offline_score_avg": round(float(loc_mismatch_df['offline_score'].mean()), 4),
            "actual_apply_rate": round(float(apply_rate), 4),
            "gap_vs_overall": round(float(overall_apply_rate - apply_rate), 4)
        })
        
    # Defect Hypothesis 2: Seniority Mismatch
    seniority_map = {'Entry Level': 1, 'Mid Level': 2, 'Senior': 3, 'Director': 4}
    high_conf_df['cand_lvl'] = high_conf_df['candidate_seniority'].map(seniority_map)
    high_conf_df['job_lvl'] = high_conf_df['job_seniority'].map(seniority_map)
    
    # Underqualified
    under_mask = high_conf_df['cand_lvl'] < high_conf_df['job_lvl']
    under_df = high_conf_df[under_mask]
    if len(under_df) > 0:
        apply_rate = under_df['is_apply'].mean()
        defects.append({
            "defect_id": "D-002",
            "name": "Underqualified Recommendation",
            "description": "Model recommends senior roles to junior candidates, resulting in no applications.",
            "impacted_volume": int(len(under_df)),
            "offline_score_avg": round(float(under_df['offline_score'].mean()), 4),
            "actual_apply_rate": round(float(apply_rate), 4),
            "gap_vs_overall": round(float(overall_apply_rate - apply_rate), 4)
        })
        
    # Overqualified
    over_mask = high_conf_df['cand_lvl'] > high_conf_df['job_lvl']
    over_df = high_conf_df[over_mask]
    if len(over_df) > 0:
        apply_rate = over_df['is_apply'].mean()
        defects.append({
            "defect_id": "D-003",
            "name": "Overqualified Recommendation",
            "description": "Model recommends junior roles to senior candidates, leading to low conversion.",
            "impacted_volume": int(len(over_df)),
            "offline_score_avg": round(float(over_df['offline_score'].mean()), 4),
            "actual_apply_rate": round(float(apply_rate), 4),
            "gap_vs_overall": round(float(overall_apply_rate - apply_rate), 4)
        })

    # Rank by impact (gap_vs_overall * impacted_volume)
    defects.sort(key=lambda x: x['gap_vs_overall'] * x['impacted_volume'], reverse=True)
    return defects

def main():
    try:
        input_path = "logs/interaction_logs.csv"
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Missing file: {input_path}")
            
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} logs for defect analysis.")
        
        defects = identify_defects(df)
        
        out_path = "logs/intelligence_defects.json"
        with open(out_path, 'w') as f:
            json.dump(defects, f, indent=4)
            
        logger.info(f"Identified {len(defects)} intelligence defects.")
        for d in defects:
            logger.info(f"Defect {d['defect_id']}: {d['name']} | Impact: {d['impacted_volume']} | Apply Rate: {d['actual_apply_rate']:.2%}")
            
        logger.info(f"Saved defects to {out_path}")
        
    except Exception as e:
        logger.critical(f"Unhandled error in defect finding: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
