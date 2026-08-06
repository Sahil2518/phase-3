import pandas as pd
import numpy as np
import logging
import uuid
import sys
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

def simulate_traffic(num_requests: int = 1000, scenario: str = "normal", random_state: int = 42) -> pd.DataFrame:
    """
    Generates synthetic API inference traffic logs to test SLOs and Error Budgets.

    The function creates realistic traffic logs with timestamps, latencies, 
    HTTP status codes, and model prediction scores. It can inject failures 
    to test the observability alerting systems.

    Parameters
    ----------
    num_requests : int
        Number of inference requests to simulate.
    scenario : str
        The traffic scenario to simulate. Options: "normal", "latency_breach", 
        "availability_breach", "quality_breach".
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    df : pd.DataFrame
        Dataframe containing the generated telemetry logs.
    """
    np.random.seed(random_state)
    logger.info(f"Simulating {num_requests} requests for scenario: '{scenario}'")
    
    # Base Timestamps (distributed over the last hour)
    base_time = datetime.now() - timedelta(hours=1)
    timestamps = [base_time + timedelta(seconds=int(s)) for s in np.random.uniform(0, 3600, num_requests)]
    timestamps.sort()

    request_ids = [str(uuid.uuid4()) for _ in range(num_requests)]
    
    # Defaults (Normal Traffic)
    # Latency: Log-normal distribution centered around 50ms, mostly < 200ms
    latencies = np.random.lognormal(mean=np.log(50), sigma=0.4, size=num_requests)
    
    # Status codes: 99.9% success
    status_probs = [0.999, 0.001]
    status_codes = np.random.choice([200, 500], size=num_requests, p=status_probs)
    
    # Scores: Normal distribution between 0.1 and 0.9
    scores = np.random.normal(loc=0.5, scale=0.15, size=num_requests)
    scores = np.clip(scores, 0.0, 1.0)
    
    if scenario == "latency_breach":
        # Inject p95 > 200ms (we make 10% of requests very slow)
        slow_indices = np.random.choice(num_requests, size=int(0.10 * num_requests), replace=False)
        latencies[slow_indices] = np.random.uniform(250, 1000, size=len(slow_indices))
        logger.warning(f"Injected latency breach: 10% of requests have high latency.")
        
    elif scenario == "availability_breach":
        # Inject 5% failure rate (Availability drops to 95%)
        fail_indices = np.random.choice(num_requests, size=int(0.05 * num_requests), replace=False)
        status_codes[fail_indices] = 503
        logger.warning(f"Injected availability breach: 5% of requests return 503 HTTP errors.")
        
    elif scenario == "quality_breach":
        # Inject silent model failure: Model returns a constant degenerate score of 0.01 for all requests
        scores = np.full(num_requests, 0.01)
        logger.warning(f"Injected quality breach: Model returning constant 0.01 degenerate score.")
    
    elif scenario != "normal":
        raise ValueError(f"Unknown scenario: {scenario}")
    
    df = pd.DataFrame({
        "timestamp": timestamps,
        "request_id": request_ids,
        "latency_ms": latencies,
        "http_status": status_codes,
        "prediction_score": scores
    })
    
    return df

def main():
    try:
        df = simulate_traffic(1000, scenario="normal")
        print(f"Generated {len(df)} logs.")
    except Exception as e:
        logger.critical(f"Traffic simulation failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
