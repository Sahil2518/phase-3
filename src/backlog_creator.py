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

def generate_backlog(defects):
    """
    Translates intelligence defects into an actionable Phase-3 backlog.

    Parameters
    ----------
    defects : list
        List of defect dictionaries.

    Returns
    -------
    backlog_md : str
        Markdown formatted string containing the backlog.
    """
    if not defects:
        logger.warning("No defects provided. Generating an empty backlog.")
        return "# Phase 3 Matching System Backlog\n\nNo defects identified. System is healthy."

    md = [
        "# Phase 3 Matching System Backlog",
        "Generated from real interaction log defect analysis.",
        ""
    ]
    
    for i, d in enumerate(defects, start=1):
        md.append(f"## {i}. Fix {d['name']} ({d['defect_id']})")
        md.append(f"**Impact:** {d['impacted_volume']} high-confidence recommendations failed to convert.")
        md.append(f"**Symptom:** {d['description']}")
        md.append(f"**Data Evidence:** The expected apply rate was implied by the offline score avg of `{d['offline_score_avg']:.2f}`, but actual conversion was `{d['actual_apply_rate']:.2%}` (Gap: `{d['gap_vs_overall']:.2%}`).")
        
        md.append("\n**Proposed Engineering Task:**")
        if d['defect_id'] == 'D-001':
            md.append("- Implement a hard-filter or severe penalty layer for cross-city on-site matches.")
            md.append("- Exclude `Remote` jobs from this penalty.")
        elif d['defect_id'] == 'D-002':
            md.append("- Implement an ordinal seniority check.")
            md.append("- Subdue match scores drastically when candidate ordinal level is strictly less than the job ordinal level.")
        elif d['defect_id'] == 'D-003':
            md.append("- Add a tunable decay penalty when candidate seniority strictly exceeds job seniority.")
        else:
            md.append("- Investigate root cause and implement an explicit guardrail in the matching engine.")
            
        md.append("\n---")
        
    return "\n".join(md)

def main():
    try:
        input_path = "logs/intelligence_defects.json"
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Missing file: {input_path}")
            
        with open(input_path, 'r') as f:
            defects = json.load(f)
            
        logger.info(f"Loaded {len(defects)} defects for backlog generation.")
        
        backlog_md = generate_backlog(defects)
        
        out_path = "logs/phase3_backlog.md"
        with open(out_path, 'w') as f:
            f.write(backlog_md)
            
        logger.info(f"Saved Phase-3 backlog to {out_path}")
        
    except Exception as e:
        logger.critical(f"Unhandled error in backlog generation: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
