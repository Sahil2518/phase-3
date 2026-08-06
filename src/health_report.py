import pandas as pd
import json
import logging
import os
import sys
from sklearn.metrics import roc_auc_score

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

def generate_health_report(df: pd.DataFrame):
    """
    Computes the offline vs online metric gap and generates a health report.

    Parameters
    ----------
    df : pd.DataFrame
        Interaction logs containing offline_score and online outcomes (is_click, is_apply).

    Returns
    -------
    report : dict
        A dictionary containing the health metrics.
    """
    if df is None or df.empty:
        raise ValueError("Input dataframe is empty or None.")
    
    # Check if required columns exist
    required_cols = ['offline_score', 'is_apply', 'is_click']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Simulated Offline baseline claims (from Phase 2 / prior validation)
    offline_auc_claim = 0.850
    offline_expected_apply_rate = df['offline_score'].mean()
    
    # Calculate actual online metrics
    # Rule 7: Guard against empty/all-zero labels for AUC
    if df['is_apply'].nunique() > 1:
        online_auc = roc_auc_score(df['is_apply'], df['offline_score'])
    else:
        logger.warning("Only one class present in y_true for AUC. Defaulting online AUC to 0.5")
        online_auc = 0.5

    actual_apply_rate = df['is_apply'].mean()
    actual_ctr = df['is_click'].mean()
    
    gap_auc = offline_auc_claim - online_auc
    gap_apply_rate = offline_expected_apply_rate - actual_apply_rate
    
    report = {
        "offline_claims": {
            "auc": round(offline_auc_claim, 4),
            "expected_apply_rate": round(offline_expected_apply_rate, 4)
        },
        "online_actuals": {
            "auc": round(online_auc, 4),
            "ctr": round(actual_ctr, 4),
            "actual_apply_rate": round(actual_apply_rate, 4)
        },
        "gaps": {
            "auc_degradation": round(gap_auc, 4),
            "apply_rate_overestimation": round(gap_apply_rate, 4)
        },
        "status": "UNHEALTHY" if gap_auc > 0.10 else "HEALTHY"
    }
    
    return report

def main():
    try:
        input_path = "logs/interaction_logs.csv"
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Missing required file: {input_path}")
        
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} interaction logs for health reporting.")
        
        report = generate_health_report(df)
        
        out_path = "logs/model_health_report.json"
        with open(out_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        logger.info("Model Health Report Generated:")
        logger.info(json.dumps(report, indent=2))
        logger.info(f"Saved health report to {out_path}")
        
    except FileNotFoundError as e:
        logger.critical(f"File Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unhandled fatal error in health reporting: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
