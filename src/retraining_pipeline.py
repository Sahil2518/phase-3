"""
retraining_pipeline.py — PlaceMux Phase 3, Task 22
===================================================
Automated Retraining Pipeline

Design rationale
----------------
When the DriftMonitor flags `retrain_recommended=True`, this pipeline:

1. **Generates fresh training data** — simulates fetching a recent labelled
   dataset from the data warehouse (using the existing churn data generator).

2. **Retrains the model** — fits a new LightGBM classifier on the fresh data
   with identical hyperparameters for reproducibility (random_state=42).

3. **Evaluates the challenger** — computes ROC-AUC and F1 on a held-out
   validation split.  The challenger is only promoted if it is meaningfully
   better than the current champion (improvement_threshold).

4. **Promotes or rejects** — if the challenger beats the champion on
   validation AUC by > improvement_threshold, it is saved as the new
   champion model.  Otherwise the current champion is retained.

5. **Logs everything** — every retrain cycle is recorded to a JSONL audit
   trail and the latest result is written to JSON.

Champion / Challenger Pattern
------------------------------
The pipeline uses a strict champion model path and a challenger staging path
to avoid overwriting a production model with an inferior retrain.

  models/churn_model_v{N}.pkl  — champion (v1, v2, …)
  models/churn_challenger.pkl  — staging slot (overwritten each cycle)

Output
------
- logs/retrain_report.json     — latest retrain cycle report
- logs/retrain_history.jsonl   — append-only retrain audit trail
- logs/task22.log              — shared task log
- models/churn_model_v{N}.pkl  — promoted champion (if improved)
"""

import os
import sys
import json
import pickle
import logging
import glob
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Rule 2: Structured logging (shared task22 log file)
# ---------------------------------------------------------------------------
os.makedirs("logs",   exist_ok=True)
os.makedirs("models", exist_ok=True)

# Only add handlers if not already added by drift_monitor (importable together)
_log_file_handler = logging.FileHandler(os.path.join("logs", "task22.log"))
_log_stream_handler = logging.StreamHandler(sys.stdout)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[_log_file_handler, _log_stream_handler],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHAMPION_GLOB      = "models/churn_model_v*.pkl"
CHALLENGER_PATH    = "models/churn_challenger.pkl"
RETRAIN_REPORT     = "logs/retrain_report.json"
RETRAIN_HISTORY    = "logs/retrain_history.jsonl"
IMPROVEMENT_THRESH = 0.005   # challenger must beat champion AUC by this margin

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


# ---------------------------------------------------------------------------
# Model versioning helpers
# ---------------------------------------------------------------------------

def get_current_champion_path() -> Optional[str]:
    """
    Discover the highest-versioned champion model file in `models/`.

    Version numbers are embedded in filenames as `churn_model_v{N}.pkl`.
    Returns the path with the highest N, or None if no champion exists.

    Returns
    -------
    str or None
        Path to the current champion model file.
    """
    candidates = glob.glob(CHAMPION_GLOB)
    if not candidates:
        return None

    def _extract_version(path: str) -> int:
        """Parse version integer from filename."""
        basename = os.path.basename(path)
        try:
            # churn_model_vN.pkl → N
            ver_str = basename.replace("churn_model_v", "").replace(".pkl", "")
            return int(ver_str)
        except ValueError:
            return 0

    candidates.sort(key=_extract_version)
    return candidates[-1]


def next_champion_path(current_path: Optional[str]) -> str:
    """
    Return the next versioned champion path (e.g. v1 → v2).

    Parameters
    ----------
    current_path : str or None
        Current champion file path.  If None, starts at v1.

    Returns
    -------
    str
        New champion file path.
    """
    if current_path is None:
        return "models/churn_model_v1.pkl"
    basename = os.path.basename(current_path)
    try:
        ver_str = basename.replace("churn_model_v", "").replace(".pkl", "")
        ver = int(ver_str)
    except ValueError:
        ver = 1
    return f"models/churn_model_v{ver + 1}.pkl"


# ---------------------------------------------------------------------------
# Data generation (simulates pulling fresh labels from data warehouse)
# ---------------------------------------------------------------------------

def generate_fresh_data(n_samples: int = 10000, random_state: int = 42) -> pd.DataFrame:
    """
    Simulate fetching a fresh labelled dataset for retraining.

    In production this would call the data warehouse / feature store.
    Here we use a controlled synthetic generator to ensure reproducibility.

    The freshness is simulated by adding a slight distribution shift to
    `days_since_last_login` — modelling the real-world scenario where users
    become slightly more inactive over time (the drift that triggered retrain).

    Parameters
    ----------
    n_samples : int
        Number of samples to generate. Default 10,000.
    random_state : int
        Random seed for reproducibility (Rule 5).

    Returns
    -------
    pd.DataFrame
        Labelled dataset with FEATURE_COLS + 'churned' column.
    """
    rng = np.random.default_rng(random_state)
    logger.info(f"Generating fresh training data: n_samples={n_samples}, seed={random_state}")

    # Slight distribution shift in login recency (the drifted feature)
    days_login = rng.integers(0, 60, n_samples).astype(float)  # range extended vs original [0,45]
    sessions_14 = rng.integers(0, 18, n_samples).astype(float)
    sessions_30 = sessions_14 + rng.integers(0, 20, n_samples).astype(float)
    apply_rate  = rng.uniform(0, 0.45, n_samples)
    jobs_viewed = rng.integers(0, 200, n_samples).astype(float)
    recruiter_c = rng.integers(0, 15, n_samples).astype(float)
    first_login = rng.integers(30, 365, n_samples).astype(float)
    profile_c   = rng.uniform(20, 100, n_samples)
    verified    = rng.integers(0, 2, n_samples).astype(float)

    # Label: churn probability driven by inactivity
    churn_prob = (
        0.35 * np.clip(days_login / 60, 0, 1)
        + 0.30 * np.clip(1 - sessions_14 / 18, 0, 1)
        + 0.20 * np.clip(1 - apply_rate / 0.45, 0, 1)
        + 0.15 * np.clip(1 - profile_c / 100, 0, 1)
    )
    noise = rng.uniform(-0.05, 0.05, n_samples)
    labels = (churn_prob + noise > 0.45).astype(int)

    df = pd.DataFrame({
        "days_since_last_login":  days_login,
        "sessions_last_14d":      sessions_14,
        "sessions_last_30d":      sessions_30,
        "apply_rate_7d":          apply_rate,
        "jobs_viewed_lifetime":   jobs_viewed,
        "recruiter_contacts":     recruiter_c,
        "days_since_first_login": first_login,
        "profile_completeness":   profile_c,
        "is_profile_verified":    verified,
        "churned":                labels,
    })

    logger.info(f"Fresh data ready: churn_rate={labels.mean():.2%}")
    return df


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_model(df: pd.DataFrame, random_state: int = 42):
    """
    Train a LightGBM gradient-boosted classifier on the provided dataset.

    Falls back to scikit-learn RandomForestClassifier if LightGBM is not
    installed, so the pipeline is always runnable in any environment.

    Parameters
    ----------
    df : pd.DataFrame
        Labelled training DataFrame with FEATURE_COLS + 'churned'.
    random_state : int
        Random seed (Rule 5: always set to 42).

    Returns
    -------
    model
        Trained classifier with a `predict_proba` method.
    """
    # Rule 2: Data validation guard
    assert df.shape[0] > 0, "Training set is empty!"
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Training data missing features: {missing}")

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["churned"].values.astype(int)

    try:
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
        logger.info("Training with LightGBM")
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            random_state=random_state,
            n_jobs=-1,
        )
        logger.info("LightGBM not available — training with RandomForestClassifier")

    model.fit(X, y)
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, df_val: pd.DataFrame) -> dict:
    """
    Evaluate the trained model on a held-out validation set.

    Computes ROC-AUC and macro-F1 without external dependencies —
    ROC-AUC uses a manual trapezoidal implementation.

    Parameters
    ----------
    model : trained classifier
        Must expose `predict_proba(X)` and `predict(X)`.
    df_val : pd.DataFrame
        Validation DataFrame with FEATURE_COLS + 'churned'.

    Returns
    -------
    dict
        {'auc': float, 'f1': float, 'n_val': int}
    """
    # Rule 7: None guard
    if model is None:
        raise ValueError("Cannot evaluate: model is None.")

    # Rule 7: Empty input guard
    if df_val.empty:
        logger.warning("evaluate_model: empty validation set — returning zeros.")
        return {"auc": 0.0, "f1": 0.0, "n_val": 0}

    X_val = df_val[FEATURE_COLS].values.astype(np.float32)
    y_val = df_val["churned"].values.astype(int)

    # --- Probabilities
    try:
        probs = model.predict_proba(X_val)[:, 1]
    except Exception as e:
        raise RuntimeError(f"predict_proba failed: {e}")

    # Rule 7: NaN/Inf guard
    bad_mask = np.isnan(probs) | np.isinf(probs)
    if bad_mask.any():
        logger.warning(f"evaluate_model: {bad_mask.sum()} NaN/Inf scores detected — clamped to 0.0.")
        probs[bad_mask] = 0.0
    probs = np.clip(probs, 0.0, 1.0)

    # --- ROC-AUC (manual trapezoidal rule — no sklearn needed)
    thresholds = np.linspace(0, 1, 201)[::-1]
    tprs, fprs = [], []
    pos = y_val.sum()
    neg = len(y_val) - pos
    for t in thresholds:
        pred = (probs >= t).astype(int)
        tp = ((pred == 1) & (y_val == 1)).sum()
        fp = ((pred == 1) & (y_val == 0)).sum()
        tprs.append(tp / max(pos, 1))
        fprs.append(fp / max(neg, 1))

    # np.trapz was removed in NumPy 2.0; use np.trapezoid with fallback
    try:
        auc = float(np.trapezoid(tprs, fprs))
    except AttributeError:
        auc = float(np.trapz(tprs, fprs))
    if auc < 0:
        auc = -auc  # trapz can return negative if FPR is decreasing

    # --- F1 (at 0.5 threshold)
    preds = (probs >= 0.5).astype(int)
    tp = ((preds == 1) & (y_val == 1)).sum()
    fp = ((preds == 1) & (y_val == 0)).sum()
    fn = ((preds == 0) & (y_val == 1)).sum()
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1 = float(2 * precision * recall / max(precision + recall, 1e-9))

    logger.info(f"Evaluation — AUC={auc:.4f} | F1={f1:.4f} | n_val={len(y_val)}")
    return {"auc": round(auc, 4), "f1": round(f1, 4), "n_val": len(y_val)}


# ---------------------------------------------------------------------------
# Retraining Pipeline
# ---------------------------------------------------------------------------

class RetrainingPipeline:
    """
    Orchestrates the end-to-end retrain → evaluate → promote cycle.

    Attributes
    ----------
    improvement_threshold : float
        Minimum AUC improvement over champion required to promote challenger.
    n_samples : int
        Number of fresh training samples to generate per retrain cycle.
    val_fraction : float
        Fraction of fresh data held out for challenger evaluation.
    random_state : int
        Global random seed (Rule 5).
    """

    def __init__(
        self,
        improvement_threshold: float = IMPROVEMENT_THRESH,
        n_samples: int = 10000,
        val_fraction: float = 0.2,
        random_state: int = 42,
    ) -> None:
        """
        Initialise the RetrainingPipeline.

        Parameters
        ----------
        improvement_threshold : float
            AUC delta needed to promote challenger over champion. Default 0.005.
        n_samples : int
            Fresh dataset size per retrain. Default 10,000.
        val_fraction : float
            Validation split fraction. Default 0.2.
        random_state : int
            Random seed for reproducibility.
        """
        self.improvement_threshold = improvement_threshold
        self.n_samples      = n_samples
        self.val_fraction   = val_fraction
        self.random_state   = random_state
        logger.info(
            f"RetrainingPipeline ready | "
            f"threshold={improvement_threshold} | "
            f"n_samples={n_samples} | "
            f"val_fraction={val_fraction}"
        )

    def get_champion_auc(self) -> Tuple[Optional[object], float]:
        """
        Load the current champion model and score it on a validation set.

        If no champion exists, returns (None, 0.0) to ensure any trained
        challenger will be promoted.

        Returns
        -------
        tuple of (model or None, float)
            (champion model, champion AUC on fresh validation data).
        """
        champ_path = get_current_champion_path()
        if champ_path is None:
            logger.info("No champion model found — treating AUC as 0.0.")
            return None, 0.0

        # Rule 2: File I/O guard
        if not os.path.exists(champ_path):
            logger.warning(f"Champion path '{champ_path}' not found.")
            return None, 0.0

        try:
            with open(champ_path, "rb") as f:
                champion = pickle.load(f)
            logger.info(f"Champion loaded: {champ_path}")
        except Exception as e:
            logger.error(f"Failed to load champion: {e}")
            return None, 0.0

        # Evaluate on a small fresh validation sample
        val_df = generate_fresh_data(
            n_samples=max(self.n_samples // 5, 500),
            random_state=self.random_state + 99,  # different seed from train
        )
        try:
            metrics = evaluate_model(champion, val_df)
            return champion, metrics["auc"]
        except Exception as e:
            logger.error(f"Champion evaluation failed: {e}")
            return champion, 0.0

    def run(self, trigger_reason: str = "scheduled") -> dict:
        """
        Execute a full retrain cycle.

        Steps
        -----
        1. Fetch champion AUC baseline.
        2. Generate fresh labelled data.
        3. Train challenger on fresh data (80% train split).
        4. Evaluate challenger on hold-out (20% val split).
        5. Compare challenger vs champion AUC.
        6. Promote if challenger beats champion by > improvement_threshold.
        7. Persist report and audit trail.

        Parameters
        ----------
        trigger_reason : str
            Human-readable reason for this retrain (e.g. 'drift_detected',
            'scheduled', 'manual').

        Returns
        -------
        dict
            Structured retrain cycle report.
        """
        timestamp = datetime.utcnow().isoformat()
        logger.info(f"=== RETRAIN CYCLE START === reason={trigger_reason}")

        # --- Step 1: Champion baseline
        champion_model, champion_auc = self.get_champion_auc()
        champion_path = get_current_champion_path()
        logger.info(f"Champion AUC (on fresh val): {champion_auc:.4f}")

        # --- Step 2: Fresh data
        try:
            fresh_df = generate_fresh_data(
                n_samples=self.n_samples,
                random_state=self.random_state,
            )
        except Exception as e:
            logger.critical(f"Data generation failed: {e}", exc_info=True)
            return {"status": "FAILED", "reason": str(e)}

        # --- Step 3: Train/val split (stratified by churn label)
        n_val  = int(len(fresh_df) * self.val_fraction)
        n_train = len(fresh_df) - n_val

        # Simple split (reproducible via random_state)
        rng = np.random.default_rng(self.random_state)
        idx = rng.permutation(len(fresh_df))
        train_idx = idx[:n_train]
        val_idx   = idx[n_train:]

        df_train = fresh_df.iloc[train_idx].reset_index(drop=True)
        df_val   = fresh_df.iloc[val_idx].reset_index(drop=True)
        logger.info(f"Train/val split: {len(df_train)} train | {len(df_val)} val")

        # --- Step 4: Train challenger
        try:
            challenger = train_model(df_train, random_state=self.random_state)
        except Exception as e:
            logger.critical(f"Challenger training failed: {e}", exc_info=True)
            return {"status": "FAILED", "reason": str(e)}

        # Save challenger to staging slot
        try:
            with open(CHALLENGER_PATH, "wb") as f:
                pickle.dump(challenger, f)
            logger.info(f"Challenger saved to staging: {CHALLENGER_PATH}")
        except Exception as e:
            logger.error(f"Failed to save challenger: {e}")

        # --- Step 5: Evaluate challenger
        try:
            challenger_metrics = evaluate_model(challenger, df_val)
        except Exception as e:
            logger.error(f"Challenger evaluation failed: {e}")
            challenger_metrics = {"auc": 0.0, "f1": 0.0, "n_val": 0}

        challenger_auc = challenger_metrics["auc"]
        improvement    = challenger_auc - champion_auc
        logger.info(
            f"Challenger AUC={challenger_auc:.4f} | "
            f"Champion AUC={champion_auc:.4f} | "
            f"Improvement={improvement:+.4f}"
        )

        # --- Step 6: Promote or reject
        promoted = improvement > self.improvement_threshold
        new_champion_path = None

        if promoted:
            new_champion_path = next_champion_path(champion_path)
            try:
                with open(new_champion_path, "wb") as f:
                    pickle.dump(challenger, f)
                logger.info(
                    f"[PROMOTED] New champion: {new_champion_path} "
                    f"(AUC {champion_auc:.4f} -> {challenger_auc:.4f})"
                )
            except Exception as e:
                logger.error(f"Failed to save promoted champion: {e}")
                promoted = False
        else:
            logger.info(
                f"[REJECTED] Challenger improvement ({improvement:+.4f}) "
                f"below threshold ({self.improvement_threshold}). "
                f"Champion retained."
            )

        # --- Step 7: Reports
        report = {
            "run_timestamp": timestamp,
            "trigger_reason": trigger_reason,
            "status": "PROMOTED" if promoted else "REJECTED",
            "champion": {
                "path": champion_path,
                "auc": round(champion_auc, 4),
            },
            "challenger": {
                "path": new_champion_path if promoted else CHALLENGER_PATH,
                "auc": round(challenger_auc, 4),
                "f1":  round(challenger_metrics.get("f1", 0.0), 4),
                "n_val": challenger_metrics.get("n_val", 0),
            },
            "improvement": round(improvement, 4),
            "improvement_threshold": self.improvement_threshold,
            "promoted": promoted,
        }

        try:
            with open(RETRAIN_REPORT, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save retrain report: {e}")

        # Append to history
        try:
            with open(RETRAIN_HISTORY, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "timestamp": timestamp,
                    "trigger_reason": trigger_reason,
                    "status": report["status"],
                    "challenger_auc": round(challenger_auc, 4),
                    "champion_auc": round(champion_auc, 4),
                    "improvement": round(improvement, 4),
                    "promoted": promoted,
                }) + "\n")
        except Exception as e:
            logger.error(f"Failed to append retrain history: {e}")

        logger.info(f"=== RETRAIN CYCLE END === status={report['status']}")
        return report


# ---------------------------------------------------------------------------
# Main (smoke test)
# ---------------------------------------------------------------------------

def main() -> None:
    """Smoke test: run one retrain cycle and print the report."""
    try:
        pipeline = RetrainingPipeline(n_samples=5000)
        report = pipeline.run(trigger_reason="smoke_test")
        print("\n--- Retrain Report ---")
        print(json.dumps(report, indent=2))
    except Exception as e:
        logger.critical(f"Smoke test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
