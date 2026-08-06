import pandas as pd
import json
import logging
import os
import sys

# Rule 2: Structured Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

def calculate_error_budget(total_requests: int, failed_requests: int, 
                           target_availability: float = 0.999) -> dict:
    """
    Calculates the remaining error budget based on allowed failures vs actual failures.

    Parameters
    ----------
    total_requests : int
        Total volume of inference requests over the window.
    failed_requests : int
        Number of 5xx failed requests.
    target_availability : float
        The SLO target for availability (e.g., 0.999 for 99.9%).

    Returns
    -------
    budget_report : dict
        A report containing error budget consumption metrics.
    """
    if total_requests <= 0:
        raise ValueError("Total requests must be > 0 to calculate error budget.")
        
    allowed_failures = int(total_requests * (1 - target_availability))
    remaining_budget = allowed_failures - failed_requests
    budget_consumed_pct = (failed_requests / allowed_failures) * 100 if allowed_failures > 0 else 100.0
    
    report = {
        "time_window": "Rolling 30-Day (Simulated)",
        "total_requests": total_requests,
        "target_availability": target_availability,
        "allowed_failures": allowed_failures,
        "actual_failures": failed_requests,
        "remaining_budget_requests": remaining_budget,
        "budget_consumed_percentage": round(budget_consumed_pct, 2),
        "status": "HEALTHY" if remaining_budget >= 0 else "DEPLETED"
    }
    
    return report

def main():
    try:
        report = calculate_error_budget(1000000, 500, 0.999)
        logger.info(f"Budget Report: {json.dumps(report, indent=2)}")
    except Exception as e:
        logger.critical(f"Error budget calculation failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
