import cProfile
import pstats
import io
import pandas as pd
import numpy as np
import logging
import os
import sys
from inference_engine import UnoptimizedInferenceEngine

# Rule 2: Structured Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def run_profiler(num_samples: int = 5000):
    """
    Runs cProfile on the unoptimized inference path to identify bottlenecks.
    
    Parameters
    ----------
    num_samples : int
        Number of synthetic records to profile against.
    """
    logger.info(f"Generating {num_samples} synthetic rows for profiling...")
    np.random.seed(42)
    df = pd.DataFrame({
        'skill_score': np.random.uniform(0, 10, num_samples),
        'experience_years': np.random.uniform(0, 15, num_samples)
    })
    
    engine = UnoptimizedInferenceEngine()
    
    logger.info("Running cProfile on UnoptimizedInferenceEngine.predict()...")
    pr = cProfile.Profile()
    pr.enable()
    
    _ = engine.predict(df)
    
    pr.disable()
    
    s = io.StringIO()
    sortby = pstats.SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(20) # Top 20 slowest calls
    
    os.makedirs("logs", exist_ok=True)
    out_path = "logs/profile_results.txt"
    with open(out_path, "w") as f:
        f.write(s.getvalue())
        
    logger.info(f"Profiling complete. Top 20 bottlenecks saved to {out_path}")

def main():
    try:
        run_profiler()
    except Exception as e:
        logger.critical(f"Profiler failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
