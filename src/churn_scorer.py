"""
churn_scorer.py — PlaceMux Phase 3, Task 8
==========================================
Loads the trained churn model and scores all *active* candidates
(those not yet definitively labelled as churned) to produce a
prioritised at-risk list for the Growth team.

Output columns
--------------
candidate_id        : unique candidate identifier
churn_score         : model predicted probability of churn (0–1)
risk_tier           : High / Medium / Low (based on fixed thresholds)
top_risk_factor     : most influential feature for this candidate
recommended_action  : plain-English intervention suggestion

Risk tier thresholds
--------------------
High   : churn_score >= 0.65  → immediate outreach
Medium : churn_score 0.40-0.65 → re-engagement campaign
Low    : churn_score < 0.40   → routine monitoring

Model-unavailability fallback
-----------------------------
If the model file cannot be loaded, the scorer falls back to a
deterministic rule-based heuristic (days_since_last_login + sessions_last_14d)
and logs a warning.  The Growth team always receives a list — never an error.
"""

import os
import sys
import json
import pickle
import logging
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Rule 2: Structured logging
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("logs", "task08.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_PATH = "models/churn_model_v1.pkl"
DATA_PATH = "logs/churn_dataset.csv"
AT_RISK_PATH = "logs/churn_at_risk_list.csv"

FEATURE_COLS = [
    "days_since_last_login",
    "sessions_last_14d",
    "sessions_last_30d",
    "apply_rate_7d",
    "jobs_viewed_lifetime",
    "recruiter_contacts",
    "days_since_first_login",
    "profile_completeness",
    "is_profile_verified",
]

# Risk tier thresholds (based on val-set calibration from train_task08.py)
TIER_HIGH = 0.65
TIER_MEDIUM = 0.40

# Recommended actions per risk tier
ACTIONS = {
    "High": "Immediate 1-to-1 re-engagement email + premium job recommendations",
    "Medium": "Automated nudge campaign: highlight new matching jobs this week",
    "Low": "Routine monitoring — include in weekly digest email",
}


# ---------------------------------------------------------------------------
# Model loading with fallback
# ---------------------------------------------------------------------------

def load_model(model_path: str = MODEL_PATH):
    """
    Load the trained churn model from disk.

    Falls back gracefully if the model file is absent or corrupt,
    returning None to signal that the heuristic scorer should be used.

    Parameters
    ----------
    model_path : str
        Path to the serialised model file.

    Returns
    -------
    model or None
        Loaded model object, or None if unavailable.
    """
    # Rule 2: File I/O guard
    if not os.path.exists(model_path):
        logger.warning(
            f"Model file not found at '{model_path}'. "
            f"Falling back to rule-based heuristic scorer."
        )
        return None

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logger.error(
            f"Failed to load model: {e}. "
            f"Falling back to rule-based heuristic scorer."
        )
        return None


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def score_with_model(model, X: np.ndarray) -> np.ndarray:
    """
    Score candidates using the trained LightGBM model.

    Rule 7: None guard, NaN/Inf output guard, empty input guard.

    Parameters
    ----------
    model : trained classifier with predict_proba
        The churn model.
    X : np.ndarray
        Feature matrix of shape (n_candidates, n_features).

    Returns
    -------
    np.ndarray
        Churn probability scores in [0, 1].
    """
    # Rule 7: None guard
    if model is None:
        raise ValueError("Cannot predict: model is uninitialized or None.")

    # Rule 7: Empty input guard
    if X.shape[0] == 0:
        logger.warning("Empty input matrix — returning empty score array.")
        return np.array([])

    raw_scores = model.predict_proba(X)[:, 1]

    # Rule 7: NaN/Inf guard
    bad_mask = np.isnan(raw_scores) | np.isinf(raw_scores)
    if bad_mask.any():
        n_bad = bad_mask.sum()
        logger.warning(
            f"{n_bad} invalid model output(s) detected (NaN/Inf). "
            f"Defaulting those to 0.0."
        )
        raw_scores[bad_mask] = 0.0

    return np.clip(raw_scores, 0.0, 1.0)


def score_with_heuristic(df: pd.DataFrame) -> np.ndarray:
    """
    Rule-based fallback scorer used when the model is unavailable.

    Heuristic: churn_score = 0.6 * (days_since_last_login / 45)
                            + 0.4 * (1 - sessions_last_14d_norm)

    This is intentionally simple — it serves as a safety net so the Growth
    team always receives an ordered list even in a model-down scenario.

    Parameters
    ----------
    df : pd.DataFrame
        Candidate feature DataFrame.

    Returns
    -------
    np.ndarray
        Heuristic churn scores in [0, 1].
    """
    logger.warning("Using rule-based heuristic scorer (model unavailable).")

    days_norm = (df["days_since_last_login"] / 45.0).clip(0, 1)
    sessions_norm = (df["sessions_last_14d"] / df["sessions_last_14d"].max().clip(1)).clip(0, 1)

    scores = 0.6 * days_norm + 0.4 * (1.0 - sessions_norm)
    return scores.clip(0.0, 1.0).values


# ---------------------------------------------------------------------------
# Risk factor explanation
# ---------------------------------------------------------------------------

def get_top_risk_factor(row: pd.Series) -> str:
    """
    Return the most influential risk signal for a single candidate in
    plain English.  Uses simple rule-based logic (not SHAP) to keep
    the scorer dependency-free and fast.

    Parameters
    ----------
    row : pd.Series
        A single candidate's feature values.

    Returns
    -------
    str
        Plain-English description of the primary risk factor.
    """
    # Priority order: recency signals > volume signals > profile signals
    if row["days_since_last_login"] >= 14:
        return f"No login for {int(row['days_since_last_login'])} days"
    if row["sessions_last_14d"] <= 1:
        return "Only 1 session in last 14 days"
    if row["apply_rate_7d"] < 0.05:
        return "Near-zero application activity this week"
    if row["profile_completeness"] < 40:
        return f"Profile only {row['profile_completeness']:.0f}% complete"
    if row["recruiter_contacts"] == 0:
        return "No recruiter contact received yet"
    return "Declining engagement trend"


# ---------------------------------------------------------------------------
# At-risk list builder
# ---------------------------------------------------------------------------

def build_at_risk_list(
    model,
    data_path: str = DATA_PATH,
    output_path: str = AT_RISK_PATH,
) -> pd.DataFrame:
    """
    Score all active candidates and return a prioritised at-risk list.

    'Active' is defined as candidates whose ground-truth label is 0
    (not yet confirmed churned) — in production this would be all
    candidates who have not been definitively disengaged.

    Parameters
    ----------
    model : classifier or None
        Trained churn model.  If None, the heuristic scorer is used.
    data_path : str
        Path to the full candidate dataset.
    output_path : str
        Where to write the at-risk CSV.

    Returns
    -------
    pd.DataFrame
        Ranked at-risk candidate list with all scoring columns.
    """
    # Rule 2: File I/O guard
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Candidate data not found at: {data_path}")

    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} candidate records for scoring.")

    # Rule 7: Empty input guard
    if df.empty:
        logger.warning("Empty candidate dataset — returning empty at-risk list.")
        return pd.DataFrame()

    # Score only active (non-churned) candidates
    active = df[df["churned"] == 0].copy().reset_index(drop=True)
    logger.info(f"Scoring {len(active)} active (non-churned) candidates.")

    X = active[FEATURE_COLS].values.astype(np.float32)

    # --- Score with model or heuristic fallback
    if model is not None:
        try:
            scores = score_with_model(model, X)
            scoring_method = "lightgbm_model"
        except Exception as e:
            logger.error(f"Model scoring failed ({e}). Falling back to heuristic.")
            scores = score_with_heuristic(active)
            scoring_method = "heuristic_fallback"
    else:
        scores = score_with_heuristic(active)
        scoring_method = "heuristic_fallback"

    active["churn_score"] = np.round(scores, 4)

    # --- Risk tier assignment
    active["risk_tier"] = pd.cut(
        active["churn_score"],
        bins=[-np.inf, TIER_MEDIUM, TIER_HIGH, np.inf],
        labels=["Low", "Medium", "High"],
    )

    # --- Plain-English top risk factor (per-row, fault-isolated)
    risk_factors = []
    for _, row in active.iterrows():
        try:
            risk_factors.append(get_top_risk_factor(row))
        except Exception as e:
            logger.warning(f"Failed to compute risk factor for candidate {row.get('candidate_id', '?')}: {e}")
            risk_factors.append("Unknown")
    active["top_risk_factor"] = risk_factors

    # --- Recommended action
    active["recommended_action"] = active["risk_tier"].map(ACTIONS)

    # --- Scoring method (for audit trail)
    active["scoring_method"] = scoring_method

    # --- Sort by descending churn score (most at-risk first)
    result = (
        active[
            [
                "candidate_id",
                "churn_score",
                "risk_tier",
                "top_risk_factor",
                "recommended_action",
                "scoring_method",
                # Include key features for analyst context
                "days_since_last_login",
                "sessions_last_14d",
                "apply_rate_7d",
                "profile_completeness",
            ]
        ]
        .sort_values("churn_score", ascending=False)
        .reset_index(drop=True)
    )

    # --- Persist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.to_csv(output_path, index=False)
    logger.info(f"At-risk list saved -> {output_path}  ({len(result)} candidates)")

    # Log tier summary
    tier_summary = result["risk_tier"].value_counts().to_dict()
    logger.info(f"Risk tier summary: {tier_summary}")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for direct script execution."""
    try:
        model = load_model()
        at_risk = build_at_risk_list(model)
        print("\n--- Top 10 At-Risk Candidates ---")
        print(at_risk.head(10).to_string(index=False))
        print("---------------------------------\n")
    except FileNotFoundError as e:
        logger.critical(f"Missing required file: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unhandled fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
