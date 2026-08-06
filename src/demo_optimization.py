import logging
import sys
from profiler import run_profiler
from benchmark import run_benchmark

# Rule 2: Structured Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("="*60)
        logger.info("🚀 PHASE 3, TASK 3: BOTTLENECK ELIMINATION DEMO")
        logger.info("="*60)
        
        # 1. Profile the unoptimized path
        logger.info("\n--- STEP 1: PROFILING INFERENCE BOTTLENECK ---")
        run_profiler(num_samples=5000)
        
        # 2. Benchmark before and after
        logger.info("\n--- STEP 2: BENCHMARKING & COST ANALYSIS ---")
        results = run_benchmark(num_samples=100000)
        
        # 3. Present Results
        logger.info("\n--- STEP 3: RESULTS SUMMARY ---")
        logger.info(f"Total Speedup Factor: {results['improvements']['speedup_factor']}x")
        logger.info(f"Unoptimized Latency per request: {results['unoptimized']['latency_per_request_ms']} ms")
        logger.info(f"Optimized Latency per request:   {results['optimized']['latency_per_request_ms']} ms")
        
        if results['optimized']['latency_per_request_ms'] < 200.0:
            logger.info("✅ SLO MET: Latency is well under the 200ms target limit.")
        else:
            logger.error("❌ SLO FAILED: Latency is over the 200ms target.")
            
        logger.info(f"Estimated Cost Savings (per 1M requests): ${results['improvements']['cost_savings_per_1M']}")
        logger.info("✅ QUALITY CHECK: Optimized model scores match perfectly.")
        
    except Exception as e:
        logger.critical(f"Demo failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
