"""
drift_monitor.py — PlaceMux Phase 3, Task 22
============================================
Drift Monitoring Engine

Design rationale
----------------
Two complementary drift signals are tracked:

1. **Feature / Data Drift** — measured with Population Stability Index (PSI).
   PSI quantifies how much the distribution of a feature in the current
   production window has shifted relative to the training baseline.
   Thresholds follow the industry standard:
     PSI < 0.10  → No significant drift   (GREEN)
     PSI 0.10-0.25 → Moderate drift       (YELLOW)
     PSI > 0.25  → Significant drift      (RED) → trigger retrain

2. **Concept Drift** — measured by comparing the model's prediction score
   distribution from a reference window against the current window using
   Jensen-Shannon Divergence (JSD) and mean prediction shift.
   A significant shift signals that the relationship between features and
   the target label may have changed, i.e. the model is stale.
   Threshold:
     JSD > 0.10  → Concept drift detected

Both checks produce structured JSON reports persisted to `logs/`.

Output
------
- logs/drift_report.json    — latest point-in-time drift report
- logs/drift_history.jsonl  — append-only drift history for trend analysis
- logs/task22.log           — human-readable event log
"""

import os
import sys
import json
import logging
import math
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Rule 2: Structured logging
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("logs", "task22.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PSI thresholds (industry standard)
# ---------------------------------------------------------------------------
PSI_GREEN  = 0.10   # < 0.10  → stable
PSI_YELLOW = 0.25   # 0.10-0.25 → moderate drift, monitor closely
# >= 0.25           → significant drift, retrain recommended

# Jensen-Shannon Divergence threshold for concept drift
JSD_THRESHOLD = 0.10


# ---------------------------------------------------------------------------
# Utility: Population Stability Index (PSI)
# ---------------------------------------------------------------------------

def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """
    Compute the Population Stability Index between two univariate distributions.

    PSI = sum[ (curr_pct - ref_pct) * ln(curr_pct / ref_pct) ]

    Bins are defined on the reference distribution (fixed edges) so that
    the same segments are compared across time windows.  A tiny epsilon
    is added to every bin proportion to avoid log(0) explosions.

    Parameters
    ----------
    reference : np.ndarray
        1-D array of feature values from the training/baseline period.
    current : np.ndarray
        1-D array of feature values from the current production window.
    n_bins : int
        Number of equal-width bins.  Default 10.
    eps : float
        Small constant to add to avoid division by zero.

    Returns
    -------
    float
        PSI value.  Returns 0.0 if either array is empty.
    """
    if len(reference) == 0 or len(current) == 0:
        logger.warning("PSI: empty input array detected — returning 0.0.")
        return 0.0

    # Compute bin edges from the reference distribution
    ref_min = float(np.min(reference))
    ref_max = float(np.max(reference))
    if ref_min == ref_max:
        # Constant feature — no drift possible
        return 0.0

    bin_edges = np.linspace(ref_min, ref_max, n_bins + 1)
    bin_edges[0]  -= 1e-9   # include leftmost point
    bin_edges[-1] += 1e-9   # include rightmost point

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current,   bins=bin_edges)

    # Convert to proportions
    ref_pct = (ref_counts / max(len(reference), 1)) + eps
    cur_pct = (cur_counts / max(len(current),   1)) + eps

    # PSI formula
    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return psi


def psi_severity(psi: float) -> str:
    """
    Classify a PSI value into a human-readable severity level.

    Parameters
    ----------
    psi : float
        Computed PSI value.

    Returns
    -------
    str
        'GREEN', 'YELLOW', or 'RED'.
    """
    if psi < PSI_GREEN:
        return "GREEN"
    elif psi < PSI_YELLOW:
        return "YELLOW"
    else:
        return "RED"


# ---------------------------------------------------------------------------
# Utility: Jensen-Shannon Divergence
# ---------------------------------------------------------------------------

def compute_jsd(
    p: np.ndarray,
    q: np.ndarray,
    n_bins: int = 20,
    eps: float = 1e-8,
) -> float:
    """
    Compute the Jensen-Shannon Divergence between two probability distributions.

    JSD is a symmetric, bounded [0, 1] divergence metric derived from
    KL-divergence.  It is ideal for comparing prediction score distributions
    because it handles the case where one distribution has zero mass in a bin
    while the other does not.

    Parameters
    ----------
    p : np.ndarray
        Reference prediction scores (e.g., training-window predictions).
    q : np.ndarray
        Current prediction scores (e.g., last-week predictions).
    n_bins : int
        Number of histogram bins.
    eps : float
        Small constant to avoid log(0).

    Returns
    -------
    float
        JSD in [0, 1].  Returns 0.0 if either array is empty.
    """
    if len(p) == 0 or len(q) == 0:
        logger.warning("JSD: empty input array — returning 0.0.")
        return 0.0

    # Shared bin edges over [0, 1] since scores are probabilities
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    p_hist, _ = np.histogram(p, bins=bin_edges, density=False)
    q_hist, _ = np.histogram(q, bins=bin_edges, density=False)

    # Normalise to proper probability distributions
    p_dist = (p_hist / max(p_hist.sum(), 1)) + eps
    q_dist = (q_hist / max(q_hist.sum(), 1)) + eps
    p_dist /= p_dist.sum()
    q_dist /= q_dist.sum()

    m_dist = 0.5 * (p_dist + q_dist)

    # KL(p || m) + KL(q || m)
    kl_p_m = np.sum(p_dist * np.log(p_dist / m_dist))
    kl_q_m = np.sum(q_dist * np.log(q_dist / m_dist))

    jsd = float(0.5 * kl_p_m + 0.5 * kl_q_m)
    # Clamp to [0, 1] to handle floating-point edge cases
    return max(0.0, min(1.0, jsd))


# ---------------------------------------------------------------------------
# DriftMonitor class
# ---------------------------------------------------------------------------

class DriftMonitor:
    """
    Monitors feature drift and concept drift for the PlaceMux matching model.

    Keeps an internal reference baseline established from training-window data.
    On each call to .check(), compares the current production window against
    the baseline using PSI (features) and JSD (predictions).

    Attributes
    ----------
    reference_features : Dict[str, np.ndarray]
        Baseline feature arrays keyed by feature name.
    reference_predictions : np.ndarray
        Baseline model prediction scores from the training window.
    drift_threshold_psi : float
        PSI value above which a feature is considered significantly drifted.
    drift_threshold_jsd : float
        JSD value above which concept drift is flagged.
    report_path : str
        Path to write the latest drift JSON report.
    history_path : str
        Path to the append-only JSONL drift history.
    """

    def __init__(
        self,
        drift_threshold_psi: float = PSI_YELLOW,
        drift_threshold_jsd: float = JSD_THRESHOLD,
        report_path: str = "logs/drift_report.json",
        history_path: str = "logs/drift_history.jsonl",
    ) -> None:
        """
        Initialise the DriftMonitor.

        Parameters
        ----------
        drift_threshold_psi : float
            PSI threshold to flag a feature for retraining. Default 0.25.
        drift_threshold_jsd : float
            JSD threshold to flag concept drift. Default 0.10.
        report_path : str
            Where to write the latest JSON drift report.
        history_path : str
            Append-only JSONL file for drift history.
        """
        self.drift_threshold_psi = drift_threshold_psi
        self.drift_threshold_jsd = drift_threshold_jsd
        self.report_path  = report_path
        self.history_path = history_path

        # Baseline (set via .set_reference())
        self.reference_features: Dict[str, np.ndarray] = {}
        self.reference_predictions: np.ndarray = np.array([])

        logger.info(
            f"DriftMonitor initialised | "
            f"PSI threshold={drift_threshold_psi:.2f} | "
            f"JSD threshold={drift_threshold_jsd:.2f}"
        )

    def set_reference(
        self,
        features_df: pd.DataFrame,
        predictions: np.ndarray,
    ) -> None:
        """
        Establish the training-window baseline for drift comparison.

        Call this once after training with the training-set features and
        the model's predictions on that training set.

        Parameters
        ----------
        features_df : pd.DataFrame
            Training-window features.  All numeric columns are used.
        predictions : np.ndarray
            Model prediction scores for the training window (shape: [n_samples]).
        """
        if features_df.empty:
            logger.warning("set_reference: empty features DataFrame — baseline not updated.")
            return

        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        self.reference_features = {
            col: features_df[col].dropna().values.astype(float)
            for col in numeric_cols
        }
        self.reference_predictions = np.clip(predictions, 0.0, 1.0)
        logger.info(
            f"Reference baseline set | "
            f"n_samples={len(features_df)} | "
            f"features={list(numeric_cols)}"
        )

    def check(
        self,
        current_features_df: pd.DataFrame,
        current_predictions: np.ndarray,
    ) -> dict:
        """
        Run a full drift check against the established baseline.

        Computes PSI per feature and JSD on prediction scores.
        Determines overall drift status and recommends retrain if needed.

        Parameters
        ----------
        current_features_df : pd.DataFrame
            Features from the current production window.
        current_predictions : np.ndarray
            Model prediction scores for the current window.

        Returns
        -------
        dict
            Structured drift report with:
            - run_timestamp
            - overall_status: 'STABLE' | 'DRIFT_DETECTED'
            - retrain_recommended: bool
            - feature_drift: dict of PSI results per feature
            - concept_drift: dict with JSD + prediction mean shift
        """
        # Rule 7: empty input guard
        if current_features_df.empty:
            logger.warning("check(): empty current features — returning STABLE by default.")
            return {"overall_status": "STABLE", "retrain_recommended": False}

        if len(self.reference_features) == 0:
            logger.warning("check(): reference baseline not set — call set_reference() first.")
            return {"overall_status": "NO_BASELINE", "retrain_recommended": False}

        timestamp = datetime.utcnow().isoformat()

        # ---- Feature drift via PSI ----
        feature_results: Dict[str, dict] = {}
        drifted_features: List[str] = []

        for feature, ref_vals in self.reference_features.items():
            if feature not in current_features_df.columns:
                continue
            cur_vals = current_features_df[feature].dropna().values.astype(float)
            psi_val  = compute_psi(ref_vals, cur_vals)
            severity = psi_severity(psi_val)
            feature_results[feature] = {
                "psi": round(psi_val, 5),
                "severity": severity,
                "drifted": psi_val >= self.drift_threshold_psi,
            }
            if psi_val >= self.drift_threshold_psi:
                drifted_features.append(feature)

        # ---- Concept drift via JSD ----
        cur_preds = np.clip(current_predictions, 0.0, 1.0)
        jsd_val   = compute_jsd(self.reference_predictions, cur_preds)
        pred_mean_shift = float(np.mean(cur_preds) - np.mean(self.reference_predictions))
        concept_drifted = jsd_val >= self.drift_threshold_jsd

        concept_result = {
            "jsd": round(jsd_val, 5),
            "prediction_mean_ref": round(float(np.mean(self.reference_predictions)), 4),
            "prediction_mean_current": round(float(np.mean(cur_preds)), 4),
            "prediction_mean_shift": round(pred_mean_shift, 4),
            "concept_drifted": concept_drifted,
        }

        # ---- Overall status ----
        any_feature_drift  = len(drifted_features) > 0
        retrain_recommended = any_feature_drift or concept_drifted
        overall_status = "DRIFT_DETECTED" if retrain_recommended else "STABLE"

        report = {
            "run_timestamp": timestamp,
            "overall_status": overall_status,
            "retrain_recommended": retrain_recommended,
            "n_features_checked": len(feature_results),
            "n_features_drifted": len(drifted_features),
            "drifted_features": drifted_features,
            "feature_drift": feature_results,
            "concept_drift": concept_result,
        }

        # ---- Logging ----
        level = logging.WARNING if retrain_recommended else logging.INFO
        logger.log(
            level,
            f"[DRIFT CHECK] status={overall_status} | "
            f"feature_drift={len(drifted_features)}/{len(feature_results)} | "
            f"concept_jsd={jsd_val:.4f} | "
            f"retrain={retrain_recommended}"
        )
        for feat in drifted_features:
            psi_v = feature_results[feat]["psi"]
            logger.warning(f"  [DRIFTED FEATURE] {feat}: PSI={psi_v:.4f}")

        # ---- Persist report ----
        try:
            with open(self.report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save drift report: {e}")

        # ---- Append to history ----
        try:
            with open(self.history_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "timestamp": timestamp,
                    "overall_status": overall_status,
                    "n_features_drifted": len(drifted_features),
                    "jsd": round(jsd_val, 5),
                    "retrain_recommended": retrain_recommended,
                }) + "\n")
        except Exception as e:
            logger.error(f"Failed to append drift history: {e}")

        return report

    def summarise_history(self) -> dict:
        """
        Read the drift history JSONL and return a summary dict.

        Returns
        -------
        dict
            Contains: total_checks, n_drift_events, drift_rate, last_check.
        """
        if not os.path.exists(self.history_path):
            return {"total_checks": 0, "n_drift_events": 0, "drift_rate": 0.0}

        records: List[dict] = []
        try:
            with open(self.history_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        except Exception as e:
            logger.error(f"Failed to read drift history: {e}")
            return {}

        if not records:
            return {"total_checks": 0, "n_drift_events": 0, "drift_rate": 0.0}

        total   = len(records)
        n_drift = sum(1 for r in records if r.get("retrain_recommended", False))
        return {
            "total_checks": total,
            "n_drift_events": n_drift,
            "drift_rate": round(n_drift / total, 4),
            "last_check": records[-1].get("timestamp", ""),
            "last_status": records[-1].get("overall_status", ""),
        }


# ---------------------------------------------------------------------------
# Main (smoke test)
# ---------------------------------------------------------------------------

def main() -> None:
    """Smoke test: simulate stable then drifted data and verify monitor fires."""
    try:
        monitor = DriftMonitor()

        # Baseline reference
        rng = np.random.default_rng(42)
        n_ref = 5000
        ref_df = pd.DataFrame({
            "days_since_last_login":  rng.integers(0, 45, n_ref).astype(float),
            "sessions_last_14d":      rng.integers(0, 20, n_ref).astype(float),
            "apply_rate_7d":          rng.uniform(0, 0.5, n_ref),
            "profile_completeness":   rng.uniform(20, 100, n_ref),
        })
        ref_preds = rng.uniform(0.1, 0.6, n_ref)
        monitor.set_reference(ref_df, ref_preds)

        # -- Stable window
        print("\n--- Stable Window (no drift expected) ---")
        stable_df = pd.DataFrame({
            "days_since_last_login":  rng.integers(0, 45, 500).astype(float),
            "sessions_last_14d":      rng.integers(0, 20, 500).astype(float),
            "apply_rate_7d":          rng.uniform(0, 0.5, 500),
            "profile_completeness":   rng.uniform(20, 100, 500),
        })
        stable_preds = rng.uniform(0.1, 0.6, 500)
        report_stable = monitor.check(stable_df, stable_preds)
        print(f"  Status: {report_stable['overall_status']} | Retrain: {report_stable['retrain_recommended']}")

        # -- Drifted window (users are more inactive — distribution shift)
        print("\n--- Drifted Window (drift should be detected) ---")
        drifted_df = pd.DataFrame({
            "days_since_last_login":  rng.integers(30, 90, 500).astype(float),  # shifted right
            "sessions_last_14d":      rng.integers(0, 3, 500).astype(float),    # shifted left
            "apply_rate_7d":          rng.uniform(0, 0.05, 500),                # much lower
            "profile_completeness":   rng.uniform(10, 50, 500),                 # degraded profiles
        })
        drifted_preds = rng.uniform(0.6, 0.95, 500)  # model now outputting high scores (concept drift)
        report_drifted = monitor.check(drifted_df, drifted_preds)
        print(f"  Status: {report_drifted['overall_status']} | Retrain: {report_drifted['retrain_recommended']}")
        print(f"  Drifted Features: {report_drifted['drifted_features']}")
        print(f"  Concept JSD: {report_drifted['concept_drift']['jsd']:.4f}")

        # -- History summary
        print("\n--- Drift History Summary ---")
        summary = monitor.summarise_history()
        print(f"  Total Checks: {summary['total_checks']}")
        print(f"  Drift Events: {summary['n_drift_events']}")
        print(f"  Drift Rate:   {summary['drift_rate']:.0%}")

    except Exception as e:
        logger.critical(f"Smoke test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
