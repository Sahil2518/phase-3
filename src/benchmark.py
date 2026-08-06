import time
import pandas as pd
import numpy as np
import logging
import json
import os
import sys
from inference_engine import UnoptimizedInferenceEngine, OptimizedInferenceEngine

# Rule 2: Structured Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def run_benchmark(num_samples: int = 100000):
    """
    Benchmarks unoptimized vs optimized inference engines for latency and estimated cost.
    
    Parameters
    ----------
    num_samples : int
        Number of synthetic records to benchmark against.
        
    Returns
    -------
    results : dict
        A dictionary containing latency and cost comparisons.
    """
    logger.info(f"Generating {num_samples} records for benchmarking...")
    np.random.seed(42)
    df = pd.DataFrame({
        'skill_score': np.random.uniform(0, 10, num_samples),
        'experience_years': np.random.uniform(0, 15, num_samples)
    })
    
    unoptimized = UnoptimizedInferenceEngine()
    optimized = OptimizedInferenceEngine()
    
    # 1. Benchmark Unoptimized
    logger.info("Benchmarking Unoptimized Engine...")
    start_time = time.time()
    unopt_scores = unoptimized.predict(df)
    unopt_duration = time.time() - start_time
    
    # 2. Benchmark Optimized
    logger.info("Benchmarking Optimized Engine...")
    start_time = time.time()
    opt_scores = optimized.predict(df)
    opt_duration = time.time() - start_time
    
    # 3. Assert Quality Equivalence
    # Using np.allclose to handle floating point variations
    if not np.allclose(unopt_scores, opt_scores, atol=1e-5):
        logger.error("Quality Mismatch! The optimized model produces different scores than the unoptimized model.")
        raise ValueError("Optimized model degraded quality.")
    else:
        logger.info("Quality Check Passed: Both engines produced identical scores.")
        
    # Cost Estimation
    # Assume 1 instance costs $1.00 per hour.
    # Cost per 1M requests = (Duration for 1M / 3600) * $1.00
    cost_per_hour = 1.00
    unopt_cost_per_1M = (unopt_duration / num_samples * 1000000) / 3600 * cost_per_hour
    opt_cost_per_1M = (opt_duration / num_samples * 1000000) / 3600 * cost_per_hour
    
    results = {
        "samples_tested": num_samples,
        "quality_match": True,
        "unoptimized": {
            "latency_total_seconds": round(unopt_duration, 4),
            "latency_per_request_ms": round((unopt_duration / num_samples) * 1000, 4),
            "est_cost_per_1M_requests": round(unopt_cost_per_1M, 4)
        },
        "optimized": {
            "latency_total_seconds": round(opt_duration, 4),
            "latency_per_request_ms": round((opt_duration / num_samples) * 1000, 4),
            "est_cost_per_1M_requests": round(opt_cost_per_1M, 4)
        },
        "improvements": {
            "speedup_factor": round(unopt_duration / (opt_duration + 1e-9), 2),
            "cost_savings_per_1M": round(unopt_cost_per_1M - opt_cost_per_1M, 4)
        }
    }
    
    os.makedirs("logs", exist_ok=True)
    out_path = "logs/benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
        
    logger.info(f"Saved benchmark results to {out_path}")
    return results

def main():
    try:
        run_benchmark()
    except Exception as e:
        logger.critical(f"Benchmarking failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
