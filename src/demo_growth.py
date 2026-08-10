import os
import sys
import json
import logging
import pandas as pd
from src.growth_simulator import GrowthSimulator

# Rule 2: Structured Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("logs", "task06_demo.log")),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_demo():
    """
    Orchestrates the end-to-end demonstration of Task 6: Growth Instrumentation.
    1. Generates normal traffic.
    2. Analyzes metrics.
    3. Traces a full funnel session.
    4. Generates broken traffic to test fault tolerance.
    """
    log_path = "logs/growth_events.jsonl"
    
    # Clean previous logs if they exist for a fresh demo
    if os.path.exists(log_path):
        os.remove(log_path)
        
    logger.info("=== Phase 3, Task 6: Growth Instrumentation Demo ===")
    
    # 1. Generate Normal Traffic
    logger.info("-> Generating 2000 simulated candidate sessions (Normal Volume)...")
    simulator = GrowthSimulator(log_path=log_path, break_mode=False)
    simulator.simulate_traffic(num_sessions=2000, items_per_impression=10)
    
    # 2. Analyze Metrics
    logger.info("-> Analyzing telemetry logs to compute offline metrics...")
    try:
        metrics_df = analyze_metrics(log_path)
        print("\n--- North-Star Metrics by Ranking Position ---")
        print(metrics_df.to_string(index=False))
        print("----------------------------------------------\n")
    except Exception as e:
        logger.error(f"Failed to analyze metrics: {e}")
        
    # 3. Trace a Full Funnel Session
    logger.info("-> Tracing a single candidate session (Impression -> Shortlist)...")
    trace_session(log_path)
    
    # 4. Break it on purpose
    logger.info("\n-> Running 'Break It On Purpose' mode (simulating missing model versions)...")
    broken_simulator = GrowthSimulator(log_path=log_path, break_mode=True)
    broken_simulator.simulate_traffic(num_sessions=50, items_per_impression=10)
    logger.info("-> Break mode completed. Check logs for gracefully handled errors.\n")

def analyze_metrics(log_path: str) -> pd.DataFrame:
    """
    Parses the JSONL log file, joins impressions with interactions, 
    and computes Click-Through Rate (CTR) and Apply Rate per position.

    Parameters
    ----------
    log_path : str
        Path to the JSONL log file.
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing the metrics per position.
    """
    # Load events
    events = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            events.append(json.loads(line.strip()))
            
    df = pd.DataFrame(events)
    
    if df.empty:
        raise ValueError("Log file is empty.")
        
    # Separate impressions and interactions
    impressions = df[df['event_type'] == 'impression'].copy()
    interactions = df[df['event_type'] != 'impression'].copy()
    
    # Explode impressions so each row is a single job impression
    impression_items = []
    for _, row in impressions.iterrows():
        imp_id = row['impression_id']
        items = row['items']
        for item in items:
            impression_items.append({
                'impression_id': imp_id,
                'job_id': item['job_id'],
                'position': item['position']
            })
            
    imp_df = pd.DataFrame(impression_items)
    
    # Aggregate impressions by position
    pos_stats = imp_df.groupby('position').size().reset_index(name='impressions')
    
    # Pivot interactions to count clicks, applies, shortlists
    if not interactions.empty:
        int_pivot = pd.pivot_table(
            interactions, 
            index=['position'], 
            columns=['event_type'], 
            values='event_id', 
            aggfunc='count',
            fill_value=0
        ).reset_index()
    else:
        int_pivot = pd.DataFrame({'position': pos_stats['position']})
        
    # Ensure columns exist
    for col in ['click', 'apply', 'shortlist']:
        if col not in int_pivot.columns:
            int_pivot[col] = 0
            
    # Join
    metrics = pd.merge(pos_stats, int_pivot, on='position', how='left').fillna(0)
    
    # Calculate Rates
    metrics['CTR (%)'] = (metrics['click'] / metrics['impressions'] * 100).round(2)
    metrics['Apply Rate (%)'] = (metrics['apply'] / metrics['impressions'] * 100).round(2)
    
    # Format output
    return metrics[['position', 'impressions', 'click', 'apply', 'shortlist', 'CTR (%)', 'Apply Rate (%)']]

def trace_session(log_path: str):
    """
    Finds and prints a trace of a session that completed a full funnel
    (impression, click, apply, shortlist).

    Parameters
    ----------
    log_path : str
        Path to the JSONL log file.
    """
    events = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            events.append(json.loads(line.strip()))
            
    df = pd.DataFrame(events)
    
    # Find an impression_id that has a shortlist event
    shortlists = df[df['event_type'] == 'shortlist']
    if shortlists.empty:
        logger.warning("No full funnel sessions found to trace.")
        return
        
    target_imp_id = shortlists.iloc[0]['impression_id']
    target_job_id = shortlists.iloc[0]['job_id']
    
    trace_events = df[(df['impression_id'] == target_imp_id)].sort_values('timestamp')
    
    print("\n--- Session Trace ---")
    print(f"Impression ID: {target_imp_id}")
    print(f"Target Job ID: {target_job_id}")
    print("-" * 21)
    
    for _, row in trace_events.iterrows():
        if row['event_type'] == 'impression':
            # Find the position of the target job
            pos = next((item['position'] for item in row['items'] if item['job_id'] == target_job_id), "Unknown")
            print(f"[{row['timestamp']}] IMPRESSION: Candidate {row['candidate_id']} viewed list. Target Job was at position {pos}.")
        else:
            if row['job_id'] == target_job_id:
                print(f"[{row['timestamp']}] {row['event_type'].upper()}: Candidate {row['event_type']}ed Job {row['job_id']} (Pos: {row['position']})")

if __name__ == "__main__":
    try:
        run_demo()
    except Exception as e:
        logger.critical(f"Unhandled fatal error: {e}", exc_info=True)
        sys.exit(1)
