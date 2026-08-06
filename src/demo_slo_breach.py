import os
import sys
import json
import logging
from traffic_simulator import simulate_traffic
from slo_monitor import SLOMonitor
from error_budget import calculate_error_budget

# Rule 2: Structured Logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/task02.log", mode='w')
    ]
)
logger = logging.getLogger(__name__)

def run_scenario(scenario_name: str, monitor: SLOMonitor):
    """
    Runs a single traffic scenario, evaluates SLOs, and computes error budgets.

    Parameters
    ----------
    scenario_name : str
        The name of the scenario to run ("normal", "latency_breach", etc.).
    monitor : SLOMonitor
        The initialized SLO Monitor instance.
    """
    logger.info("=" * 60)
    logger.info(f"🚀 RUNNING SCENARIO: {scenario_name.upper()}")
    logger.info("=" * 60)
    
    # 1. Simulate Traffic
    df = simulate_traffic(num_requests=5000, scenario=scenario_name)
    
    # 2. Evaluate SLOs
    report = monitor.evaluate(df)
    
    # 3. Calculate Error Budget (Assuming a larger rolling window context for the budget)
    # We'll treat this 5000 request batch as a slice of a 100,000 request window for budget purposes
    total_window_requests = 100000
    # Base failures if it was healthy would be ~10. Add the failures from our current slice.
    failed_in_slice = len(df[df['http_status'] >= 500])
    total_failures = 50 + failed_in_slice # 50 prior failures in the window
    
    budget_report = calculate_error_budget(
        total_requests=total_window_requests,
        failed_requests=total_failures,
        target_availability=monitor.target_availability
    )
    
    # Output Results
    logger.info("--- SLO Evaluation Results ---")
    metrics = report['metrics']
    logger.info(f"Availability: {metrics['availability']:.4%} (Target: >={monitor.target_availability:.1%})")
    logger.info(f"p95 Latency:  {metrics['p95_latency']:.2f}ms (Target: <={monitor.target_p95_latency}ms)")
    logger.info(f"Score Var:    {metrics['score_variance']:.6f} (Target: >={monitor.min_score_variance})")
    
    if report['alerts_fired']:
        logger.error(f"🚨 ALERTS FIRED ({len(report['alerts'])}):")
        for alert in report['alerts']:
            logger.error(f"   [{alert['type']}] {alert['message']}")
    else:
        logger.info("✅ ALL SLOS MET. NO ALERTS.")
        
    logger.info("--- Error Budget Status ---")
    logger.info(f"Remaining Budget: {budget_report['remaining_budget_requests']} requests")
    logger.info(f"Budget Consumed:  {budget_report['budget_consumed_percentage']}%")
    logger.info(f"Status:           {budget_report['status']}")
    
    # Save the reports
    out_file = f"logs/slo_report_{scenario_name}.json"
    full_report = {
        "scenario": scenario_name,
        "slo_evaluation": report,
        "error_budget": budget_report
    }
    with open(out_file, "w") as f:
        json.dump(full_report, f, indent=4)
        
    logger.info(f"Saved scenario report to {out_file}\n")


def main():
    try:
        monitor = SLOMonitor(target_availability=0.999, target_p95_latency=200)
        
        # Scenario 1: Normal healthy traffic
        run_scenario("normal", monitor)
        
        # Scenario 2: Latency spike (e.g. database slowdown)
        run_scenario("latency_breach", monitor)
        
        # Scenario 3: Availability drop (e.g. model container crashing)
        run_scenario("availability_breach", monitor)
        
        # Scenario 4: Quality drop (e.g. model degraded, returning constant scores)
        run_scenario("quality_breach", monitor)
        
    except Exception as e:
        logger.critical(f"Demo failed with error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
