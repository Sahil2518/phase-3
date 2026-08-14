"""
holdout_manager.py -- PlaceMux Phase 3, Task 9
===============================================
Permanent holdout group management and cumulative model value measurement.

Purpose
-------
A permanent holdout is a fixed slice of users (10%) who are never exposed
to any experiment.  By comparing their long-run metrics to the treatment
and control groups, we can measure cumulative model value -- the total lift
that the current generation of models provides over the no-model baseline.

Key invariant
-------------
The holdout assignment is keyed on user_id ONLY (no experiment_id).
This means a user in holdout stays there for ALL experiments, forever.
This is the only way to get an uncontaminated long-run baseline.

Metrics computed
----------------
- CTR (click-through rate) per group
- Apply rate per group
- Cumulative lift: (treatment_metric - holdout_metric) / holdout_metric
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Rule 2: Structured logging
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("logs", "task09.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HOLDOUT_REPORT_PATH = "logs/holdout_report.json"
EVENTS_LOG_PATH = "logs/experiment_events.jsonl"


# ---------------------------------------------------------------------------
# Holdout Manager
# ---------------------------------------------------------------------------

class HoldoutManager:
    """
    Manages the permanent holdout group and computes cumulative lift metrics.

    The holdout group is identified by the 'holdout' variant label in the
    experiment assignment log.  This class aggregates event logs by variant
    and computes the metrics needed to measure long-run model value.

    Usage
    -----
    manager = HoldoutManager(events_log_path="logs/experiment_events.jsonl")
    report = manager.compute_cumulative_lift()
    """

    def __init__(self, events_log_path: str = EVENTS_LOG_PATH) -> None:
        """
        Initialise the HoldoutManager.

        Parameters
        ----------
        events_log_path : str
            Path to the experiment events JSONL log produced by the simulator.
        """
        self.events_log_path = events_log_path
        logger.info(f"HoldoutManager initialised. Events log: {events_log_path}")

    def load_events(self) -> pd.DataFrame:
        """
        Load and parse the experiment events log.

        Each line is a JSON event with at minimum:
          user_id, variant, event_type ('impression', 'click', 'apply'),
          experiment_id, timestamp.

        Returns
        -------
        pd.DataFrame
            Events dataframe, or empty DataFrame if log is missing/empty.
        """
        # Rule 2: File I/O guard
        if not os.path.exists(self.events_log_path):
            logger.warning(
                f"Events log not found at '{self.events_log_path}'. "
                f"Returning empty DataFrame."
            )
            return pd.DataFrame()

        events = []
        with open(self.events_log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed event line: {e}")

        if not events:
            logger.warning("Events log is empty.")
            return pd.DataFrame()

        df = pd.DataFrame(events)
        logger.info(f"Loaded {len(df)} events from {self.events_log_path}")
        return df

    def compute_group_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute per-variant CTR and apply rate from event logs.

        Methodology
        -----------
        - Count impressions, clicks, and applies per variant.
        - CTR = clicks / impressions
        - Apply rate = applies / impressions
        - Groups: control, treatment, holdout

        Parameters
        ----------
        df : pd.DataFrame
            Events dataframe with columns: variant, event_type.

        Returns
        -------
        pd.DataFrame
            Per-variant metrics table.
        """
        # Rule 7: Empty input guard
        if df.empty:
            logger.warning("Empty events dataframe -- returning empty metrics.")
            return pd.DataFrame()

        required_cols = {"variant", "event_type"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Events dataframe missing columns: {missing}")

        groups = df["variant"].unique()
        rows = []

        for variant in groups:
            vdf = df[df["variant"] == variant]
            impressions = (vdf["event_type"] == "impression").sum()
            clicks = (vdf["event_type"] == "click").sum()
            applies = (vdf["event_type"] == "apply").sum()

            ctr = clicks / impressions if impressions > 0 else 0.0
            apply_rate = applies / impressions if impressions > 0 else 0.0

            rows.append({
                "variant": variant,
                "impressions": int(impressions),
                "clicks": int(clicks),
                "applies": int(applies),
                "ctr": round(float(ctr), 4),
                "apply_rate": round(float(apply_rate), 4),
            })

        return pd.DataFrame(rows).sort_values("variant")

    def compute_cumulative_lift(self) -> dict:
        """
        Compute cumulative lift of control and treatment over the holdout baseline.

        Lift formula
        ------------
        lift = (variant_metric - holdout_metric) / holdout_metric

        A positive lift means the variant outperforms the no-model holdout baseline.
        This is the true measure of cumulative model value over time.

        Returns
        -------
        dict
            Report containing per-variant metrics and cumulative lift vs holdout.
        """
        df = self.load_events()

        if df.empty:
            return {"error": "No events data available."}

        metrics = self.compute_group_metrics(df)

        if "holdout" not in metrics["variant"].values:
            logger.warning(
                "No holdout events found. Cumulative lift cannot be computed. "
                "Ensure holdout users generate events in the simulator."
            )
            report = {
                "run_timestamp": datetime.utcnow().isoformat(),
                "status": "INCOMPLETE",
                "reason": "No holdout group events found.",
                "metrics": metrics.to_dict(orient="records"),
            }
            return report

        holdout_row = metrics[metrics["variant"] == "holdout"].iloc[0]
        holdout_ctr = holdout_row["ctr"]
        holdout_apply = holdout_row["apply_rate"]

        lift_rows = []
        for _, row in metrics.iterrows():
            if row["variant"] == "holdout":
                ctr_lift = 0.0
                apply_lift = 0.0
            else:
                ctr_lift = (
                    (row["ctr"] - holdout_ctr) / holdout_ctr
                    if holdout_ctr > 0 else 0.0
                )
                apply_lift = (
                    (row["apply_rate"] - holdout_apply) / holdout_apply
                    if holdout_apply > 0 else 0.0
                )

            lift_rows.append({
                "variant": row["variant"],
                "impressions": row["impressions"],
                "ctr": row["ctr"],
                "apply_rate": row["apply_rate"],
                "ctr_lift_vs_holdout": round(float(ctr_lift), 4),
                "apply_lift_vs_holdout": round(float(apply_lift), 4),
            })

        report = {
            "run_timestamp": datetime.utcnow().isoformat(),
            "status": "OK",
            "holdout_baseline": {
                "ctr": float(holdout_ctr),
                "apply_rate": float(holdout_apply),
                "impressions": int(holdout_row["impressions"]),
            },
            "variant_lifts": lift_rows,
        }

        # Persist report
        try:
            with open(HOLDOUT_REPORT_PATH, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Holdout report saved -> {HOLDOUT_REPORT_PATH}")
        except Exception as e:
            logger.error(f"Failed to save holdout report: {e}")

        return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Quick test of holdout metric computation."""
    try:
        manager = HoldoutManager()
        report = manager.compute_cumulative_lift()
        print(json.dumps(report, indent=2))
    except Exception as e:
        logger.critical(f"Unhandled fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
