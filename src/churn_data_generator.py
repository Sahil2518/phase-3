"""
churn_data_generator.py — PlaceMux Phase 3, Task 8
====================================================
Generates a synthetic but realistic candidate-level cohort dataset for churn
prediction.  All features are computed from a 45-day observation window;
the churn label is derived from activity in days 46-60.  This strict temporal
separation guarantees zero label leakage.

Anti-leakage contract
---------------------
  Feature window : days 1-45  (what we know at scoring time)
  Label window   : days 46-60 (the future we are predicting)
  Operating point: score at day 45; intervene in days 46-60.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime

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
# Public API
# ---------------------------------------------------------------------------

def generate_churn_dataset(
    n_candidates: int = 3000,
    churn_rate_target: float = 0.30,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate a candidate-level cohort DataFrame suitable for churn modelling.

    Design rationale
    ----------------
    We simulate 60 days of candidate behaviour on the PlaceMux platform.
    At day 45 we freeze features; the churn label is whether the candidate
    was **completely inactive** in days 46-60.  Features are engineered to
    be realistic predictors without leaking future information.

    Parameters
    ----------
    n_candidates : int
        Number of synthetic candidate records to generate.
    churn_rate_target : float
        Approximate fraction of candidates who will be labelled as churned.
        The actual rate may vary slightly due to stochastic generation.
    random_state : int
        Seed for reproducibility (Rule 5).

    Returns
    -------
    pd.DataFrame
        Labelled dataset with candidate-level features and churn label.
        Columns: candidate_id, <feature columns>, churned (0/1).
    """
    # Rule 5: reproducibility
    rng = np.random.default_rng(random_state)
    logger.info(
        f"Generating churn dataset: n={n_candidates}, "
        f"target_churn_rate={churn_rate_target:.0%}, seed={random_state}"
    )

    # ------------------------------------------------------------------
    # 1. Core candidate archetypes (drives engagement level)
    # ------------------------------------------------------------------
    # Three latent engagement archetypes: engaged, at-risk, dormant
    archetype_probs = [0.50, 0.30, 0.20]  # engaged / at-risk / dormant
    archetypes = rng.choice(["engaged", "at_risk", "dormant"], size=n_candidates, p=archetype_probs)

    # ------------------------------------------------------------------
    # 2. Feature simulation (observation window: days 1-45)
    # ------------------------------------------------------------------
    # Profile completeness (0-100%) — higher in engaged users
    profile_completeness = np.where(
        archetypes == "engaged",
        rng.beta(8, 2, n_candidates) * 100,
        np.where(
            archetypes == "at_risk",
            rng.beta(4, 4, n_candidates) * 100,
            rng.beta(2, 8, n_candidates) * 100,
        ),
    ).clip(0, 100)

    # Days since last login (lower = more engaged)
    days_since_last_login = np.where(
        archetypes == "engaged",
        rng.integers(0, 5, n_candidates),
        np.where(
            archetypes == "at_risk",
            rng.integers(5, 20, n_candidates),
            rng.integers(15, 46, n_candidates),
        ),
    ).astype(float)

    # Sessions in last 14 days (recency window)
    sessions_last_14d = np.where(
        archetypes == "engaged",
        rng.integers(5, 30, n_candidates),
        np.where(
            archetypes == "at_risk",
            rng.integers(1, 10, n_candidates),
            rng.integers(0, 3, n_candidates),
        ),
    ).astype(float)

    # Sessions in last 30 days (medium-term window)
    sessions_last_30d = (
        sessions_last_14d
        + np.where(
            archetypes == "engaged",
            rng.integers(5, 20, n_candidates),
            np.where(
                archetypes == "at_risk",
                rng.integers(0, 8, n_candidates),
                rng.integers(0, 2, n_candidates),
            ),
        )
    ).astype(float)

    # Apply rate last 7 days (applications per session)
    apply_rate_7d = np.where(
        archetypes == "engaged",
        rng.uniform(0.2, 0.8, n_candidates),
        np.where(
            archetypes == "at_risk",
            rng.uniform(0.0, 0.3, n_candidates),
            rng.uniform(0.0, 0.05, n_candidates),
        ),
    )

    # Total jobs viewed over lifetime on platform (days 1-45)
    jobs_viewed_lifetime = np.where(
        archetypes == "engaged",
        rng.integers(30, 200, n_candidates),
        np.where(
            archetypes == "at_risk",
            rng.integers(5, 60, n_candidates),
            rng.integers(0, 15, n_candidates),
        ),
    ).astype(float)

    # Recruiter contacts received (days 1-45)
    recruiter_contacts = np.where(
        archetypes == "engaged",
        rng.integers(2, 20, n_candidates),
        np.where(
            archetypes == "at_risk",
            rng.integers(0, 5, n_candidates),
            rng.integers(0, 2, n_candidates),
        ),
    ).astype(float)

    # Days since first login (platform tenure)
    days_since_first_login = np.where(
        archetypes == "engaged",
        rng.integers(20, 45, n_candidates),
        np.where(
            archetypes == "at_risk",
            rng.integers(10, 45, n_candidates),
            rng.integers(1, 30, n_candidates),
        ),
    ).astype(float)

    # Whether the candidate has a verified profile (binary)
    is_profile_verified = rng.choice(
        [0, 1],
        size=n_candidates,
        p=[0.3, 0.7] if True else [0.5, 0.5],
    )
    # Engaged users more likely to be verified
    is_profile_verified = np.where(
        archetypes == "engaged",
        rng.choice([0, 1], size=n_candidates, p=[0.1, 0.9]),
        np.where(
            archetypes == "at_risk",
            rng.choice([0, 1], size=n_candidates, p=[0.4, 0.6]),
            rng.choice([0, 1], size=n_candidates, p=[0.7, 0.3]),
        ),
    )

    # ------------------------------------------------------------------
    # 3. Churn label (label window: days 46-60) — NO leakage
    # ------------------------------------------------------------------
    # We compute churn probability from the archetype + some features,
    # then sample the binary label from that probability.
    # The label is determined by future behaviour, not the features directly.
    base_churn_prob = np.where(
        archetypes == "engaged", 0.05,
        np.where(archetypes == "at_risk", 0.45, 0.85)
    ).astype(float)

    # Adjust slightly for days_since_last_login (the closer to 45, the higher churn risk)
    login_factor = (days_since_last_login / 45.0) * 0.2
    churn_prob = (base_churn_prob + login_factor).clip(0.0, 1.0)

    # Sample binary churn label
    churned = rng.binomial(1, churn_prob)

    # ------------------------------------------------------------------
    # 4. Assemble DataFrame
    # ------------------------------------------------------------------
    df = pd.DataFrame(
        {
            "candidate_id": [f"cand_{i:05d}" for i in range(n_candidates)],
            "archetype": archetypes,  # kept for diagnostics, NOT a model feature
            "days_since_last_login": days_since_last_login.round(0).astype(int),
            "sessions_last_14d": sessions_last_14d.astype(int),
            "sessions_last_30d": sessions_last_30d.astype(int),
            "apply_rate_7d": apply_rate_7d.round(4),
            "jobs_viewed_lifetime": jobs_viewed_lifetime.astype(int),
            "recruiter_contacts": recruiter_contacts.astype(int),
            "days_since_first_login": days_since_first_login.astype(int),
            "profile_completeness": profile_completeness.round(1),
            "is_profile_verified": is_profile_verified.astype(int),
            "churned": churned,
        }
    )

    actual_rate = churned.mean()
    logger.info(
        f"Dataset generated: {n_candidates} rows, "
        f"churn_rate={actual_rate:.2%}, "
        f"archetype_dist={pd.Series(archetypes).value_counts().to_dict()}"
    )

    # Rule 2: Data validation guards
    assert df.shape[0] > 0, "Generated dataset is empty!"
    assert not df.isnull().all(axis=1).any(), "Rows with all-NaN values found."
    assert df["churned"].nunique() == 2, "Churn label has only one class — label generation failed."

    return df


def save_churn_dataset(df: pd.DataFrame, path: str = "logs/churn_dataset.csv") -> None:
    """
    Persist the churn dataset to disk with validation.

    Parameters
    ----------
    df : pd.DataFrame
        The labelled dataset to save.
    path : str
        Output file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Churn dataset saved -> {path}  ({df.shape[0]} rows, {df.shape[1]} cols)")


def main() -> None:
    """Entry point: generate and save the churn dataset."""
    try:
        df = generate_churn_dataset()
        save_churn_dataset(df)
    except AssertionError as e:
        logger.critical(f"Data validation failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unhandled fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
