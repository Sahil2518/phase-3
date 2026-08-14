"""
train_task08.py — PlaceMux Phase 3, Task 8
==========================================
Churn prediction model training pipeline.

Stages
------
A. Baseline: random classifier (AUPRC ≈ churn_rate — the bar to beat)
B. LightGBM binary classifier with AUPRC as the primary metric
C. Honest evaluation on a held-out test set:
   - PR curve data (precision, recall, threshold at every point)
   - Lift@Top10%
   - Precision/Recall at F1-optimal threshold
D. All results written to logs/ as JSON for full reproducibility

Reproducibility guarantee (Rule 5): random_state=42 everywhere seeds accepted.
"""

import os
import sys
import json
import logging
import pickle
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
# Constants — all tuneable in one place
# ---------------------------------------------------------------------------
DATA_PATH = "logs/churn_dataset.csv"
MODEL_PATH = "models/churn_model_v1.pkl"
METRICS_PATH = "logs/task08_metrics.json"
PR_CURVE_PATH = "logs/task08_pr_curve.json"
RANDOM_STATE = 42

# Feature columns (archetype is diagnostic only — excluded from model)
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
LABEL_COL = "churned"


# ---------------------------------------------------------------------------
# Step 1: Load & split data
# ---------------------------------------------------------------------------

def load_and_split(data_path: str):
    """
    Load the churn dataset and produce stratified train/val/test splits.

    Split rationale: 60/20/20 stratified on the churn label to preserve
    class proportions in every fold.  Threshold is tuned on val; final
    numbers are reported on test only — no double-dipping.

    Parameters
    ----------
    data_path : str
        Path to the churn dataset CSV.

    Returns
    -------
    tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        Feature matrices and label vectors for each split.
    """
    # Rule 2: File I/O guard
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Churn dataset not found at: {data_path}")

    df = pd.read_csv(data_path)
    logger.info(f"Loaded dataset: {df.shape[0]} rows, churn_rate={df[LABEL_COL].mean():.2%}")

    # Rule 2: Data validation
    assert df.shape[0] >= 100, "Dataset too small for meaningful training."
    assert LABEL_COL in df.columns, f"Label column '{LABEL_COL}' missing."
    for col in FEATURE_COLS:
        assert col in df.columns, f"Feature column '{col}' missing."

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[LABEL_COL].values.astype(int)

    # Stratified 60/20/20 split using sklearn
    from sklearn.model_selection import train_test_split

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=RANDOM_STATE
    )

    logger.info(
        f"Split: train={len(y_train)} "
        f"(churn={y_train.mean():.2%}), "
        f"val={len(y_val)} (churn={y_val.mean():.2%}), "
        f"test={len(y_test)} (churn={y_test.mean():.2%})"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Step 2: Baseline (random classifier)
# ---------------------------------------------------------------------------

def evaluate_baseline(y_test: np.ndarray) -> dict:
    """
    Evaluate a random (majority-class frequency) baseline classifier.

    For imbalanced binary classification the random baseline AUPRC equals
    the prevalence of the positive class (churn rate).  Beating this bar is
    the minimum requirement.

    Parameters
    ----------
    y_test : np.ndarray
        True labels for the test set.

    Returns
    -------
    dict
        Baseline AUPRC, AUROC, and churn prevalence.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    churn_rate = y_test.mean()
    # Random scores uniformly distributed — equivalent to random ranking
    np.random.seed(RANDOM_STATE)
    random_scores = np.random.uniform(0, 1, len(y_test))

    baseline_auprc = average_precision_score(y_test, random_scores)
    baseline_auroc = roc_auc_score(y_test, random_scores)

    logger.info(
        f"[BASELINE] AUPRC={baseline_auprc:.4f} "
        f"(~= churn_rate={churn_rate:.4f}), "
        f"AUROC={baseline_auroc:.4f} (~= 0.50)"
    )

    return {
        "churn_rate": float(churn_rate),
        "baseline_auprc": float(baseline_auprc),
        "baseline_auroc": float(baseline_auroc),
    }


# ---------------------------------------------------------------------------
# Step 3: Train LightGBM model
# ---------------------------------------------------------------------------

def train_lightgbm(X_train, y_train, X_val, y_val):
    """
    Train a LightGBM binary classifier, using early stopping on the
    validation set to prevent overfitting.

    Design decisions
    ----------------
    - `scale_pos_weight` balances the class imbalance (neg/pos ratio).
    - Metric = 'average_precision' which directly optimises AUPRC.
    - 500 max rounds with early stopping at 30 rounds prevents overfit.
    - All hyperparameters are explicit (no silent defaults) for auditability.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training labels.
    X_val : np.ndarray
        Validation feature matrix (used for early stopping only).
    y_val : np.ndarray
        Validation labels.

    Returns
    -------
    model : lightgbm.LGBMClassifier
        Trained LightGBM classifier.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError(
            "LightGBM not installed.  Run: pip install lightgbm"
        )

    # Handle class imbalance: weight = neg_count / pos_count
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / max(pos_count, 1)
    logger.info(
        f"[LGBM] Class distribution: neg={neg_count}, pos={pos_count}, "
        f"scale_pos_weight={scale_pos_weight:.2f}"
    )

    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )

    logger.info(
        f"[LGBM] Training complete. Best iteration: {model.best_iteration_}"
    )
    return model


# ---------------------------------------------------------------------------
# Step 4: Honest evaluation (test set only)
# ---------------------------------------------------------------------------

def evaluate_model(model, X_val, y_val, X_test, y_test) -> dict:
    """
    Evaluate the trained model on the held-out test set.

    Threshold policy: the F1-optimal threshold is determined on the
    *validation set* and then applied to the *test set*.  This prevents
    the threshold from being tuned on test data.

    Metrics computed
    ----------------
    - AUPRC on test set
    - AUROC on test set
    - Precision / Recall / F1 at val-optimised threshold
    - Lift@Top10% on test set

    Parameters
    ----------
    model : trained classifier with predict_proba
    X_val : np.ndarray
        Validation features (for threshold tuning only).
    y_val : np.ndarray
        Validation labels.
    X_test : np.ndarray
        Test features (for reporting only).
    y_test : np.ndarray
        Test labels (the ground truth we report against).

    Returns
    -------
    dict
        Full metrics dictionary with all evaluation results.
    """
    from sklearn.metrics import (
        average_precision_score,
        roc_auc_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        f1_score,
    )

    # --- Scores on val (threshold selection) and test (reporting)
    val_scores = model.predict_proba(X_val)[:, 1]
    test_scores = model.predict_proba(X_test)[:, 1]

    # Rule 7: NaN/Inf output guard
    if np.any(np.isnan(test_scores)) or np.any(np.isinf(test_scores)):
        logger.warning("Model produced NaN/Inf scores — clamping to [0, 1].")
        test_scores = np.nan_to_num(test_scores, nan=0.0, posinf=1.0, neginf=0.0)

    # --- Core metrics
    test_auprc = average_precision_score(y_test, test_scores)
    test_auroc = roc_auc_score(y_test, test_scores)
    logger.info(f"[EVAL TEST] AUPRC={test_auprc:.4f}, AUROC={test_auroc:.4f}")

    # --- F1-optimal threshold on val set
    prec_val, rec_val, thresh_val = precision_recall_curve(y_val, val_scores)
    # Avoid divide-by-zero in F1
    f1_vals = np.where(
        (prec_val + rec_val) > 0,
        2 * prec_val * rec_val / (prec_val + rec_val + 1e-9),
        0.0,
    )
    best_idx = np.argmax(f1_vals[:-1])  # last element has no threshold
    best_threshold = float(thresh_val[best_idx])
    logger.info(
        f"[THRESHOLD] F1-optimal on val: threshold={best_threshold:.4f}, "
        f"val_F1={f1_vals[best_idx]:.4f}"
    )

    # --- Apply threshold to test set
    y_pred = (test_scores >= best_threshold).astype(int)
    test_precision = precision_score(y_test, y_pred, zero_division=0)
    test_recall = recall_score(y_test, y_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_pred, zero_division=0)
    logger.info(
        f"[EVAL TEST @ thr={best_threshold:.3f}] "
        f"Precision={test_precision:.4f}, Recall={test_recall:.4f}, F1={test_f1:.4f}"
    )

    # --- Lift@Top10%
    lift_at_10 = compute_lift_at_k(y_test, test_scores, k_fraction=0.10)
    logger.info(f"[LIFT] Lift@Top10%={lift_at_10:.2f}x")

    # --- PR curve data (for plotting / reporting)
    prec_test, rec_test, thresh_test = precision_recall_curve(y_test, test_scores)
    pr_curve = {
        "precision": prec_test.tolist(),
        "recall": rec_test.tolist(),
        "thresholds": thresh_test.tolist(),
        "auprc": float(test_auprc),
    }

    metrics = {
        "run_timestamp": datetime.utcnow().isoformat(),
        "model": "LightGBM",
        "split_sizes": {
            "val": int(len(y_val)),
            "test": int(len(y_test)),
        },
        "test_auprc": float(test_auprc),
        "test_auroc": float(test_auroc),
        "operating_threshold": best_threshold,
        "test_precision_at_threshold": float(test_precision),
        "test_recall_at_threshold": float(test_recall),
        "test_f1_at_threshold": float(test_f1),
        "lift_at_top10pct": float(lift_at_10),
    }

    return metrics, pr_curve


def compute_lift_at_k(y_true: np.ndarray, scores: np.ndarray, k_fraction: float = 0.10) -> float:
    """
    Compute the lift in the top-k% of predicted risk scores.

    Lift = (true_positives_in_top_k / total_in_top_k) / overall_churn_rate

    A lift of 2.0x means the top-10% list contains twice the fraction of
    churners compared to a random list of the same size.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels.
    scores : np.ndarray
        Predicted churn probability scores.
    k_fraction : float
        Fraction of the dataset to consider as "top-k".

    Returns
    -------
    float
        Lift value (≥ 1.0 means better than random).
    """
    n = len(y_true)
    k = max(1, int(n * k_fraction))

    # Sort by descending score and take top-k
    top_k_indices = np.argsort(scores)[::-1][:k]
    top_k_labels = y_true[top_k_indices]

    precision_at_k = top_k_labels.mean()
    overall_rate = y_true.mean()

    if overall_rate == 0:
        return 1.0  # degenerate case

    return float(precision_at_k / overall_rate)


# ---------------------------------------------------------------------------
# Step 5: Persist model and metrics
# ---------------------------------------------------------------------------

def save_model(model, path: str = MODEL_PATH) -> None:
    """
    Serialise the trained model to disk using pickle.

    Parameters
    ----------
    model : any sklearn-compatible model
        Trained model object.
    path : str
        Destination file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved -> {path}")


def save_metrics(metrics: dict, path: str = METRICS_PATH) -> None:
    """
    Write evaluation metrics to a JSON file for reproducibility (Rule 5).

    Parameters
    ----------
    metrics : dict
        Metrics dictionary to persist.
    path : str
        Output JSON path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved -> {path}")


def save_pr_curve(pr_curve: dict, path: str = PR_CURVE_PATH) -> None:
    """
    Write PR curve data to a JSON file for offline analysis.

    Parameters
    ----------
    pr_curve : dict
        Dictionary with 'precision', 'recall', 'thresholds', 'auprc'.
    path : str
        Output JSON path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pr_curve, f, indent=2)
    logger.info(f"PR curve data saved -> {path}")


# ---------------------------------------------------------------------------
# Step 6: Feature importance (explainability)
# ---------------------------------------------------------------------------

def get_feature_importance(model) -> pd.DataFrame:
    """
    Extract and return feature importances from the trained LightGBM model.

    Uses the 'gain' importance type, which measures the total gain across all
    splits where the feature is used — more robust than 'split' count.

    Parameters
    ----------
    model : lgb.LGBMClassifier
        Trained LightGBM model.

    Returns
    -------
    pd.DataFrame
        Feature names and their importance scores, sorted descending.
    """
    importances = model.booster_.feature_importance(importance_type="gain")
    fi_df = pd.DataFrame(
        {"feature": FEATURE_COLS, "importance": importances}
    ).sort_values("importance", ascending=False)
    return fi_df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_training_pipeline() -> tuple:
    """
    End-to-end training pipeline.  Generates data if needed, trains,
    evaluates, saves model and metrics.

    Returns
    -------
    tuple
        (model, metrics, pr_curve, feature_importance_df)
    """
    logger.info("=" * 60)
    logger.info("Task 8 — Churn Prediction Training Pipeline")
    logger.info("=" * 60)

    # -- Generate data if not already present
    if not os.path.exists(DATA_PATH):
        logger.info("Churn dataset not found — generating now...")
        from src.churn_data_generator import generate_churn_dataset, save_churn_dataset
        df = generate_churn_dataset()
        save_churn_dataset(df, DATA_PATH)

    # -- Load & split
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split(DATA_PATH)

    # -- Baseline
    baseline = evaluate_baseline(y_test)

    # -- Train model
    logger.info("[TRAIN] Fitting LightGBM classifier...")
    model = train_lightgbm(X_train, y_train, X_val, y_val)

    # -- Evaluate (honest — test set only for reporting)
    metrics, pr_curve = evaluate_model(model, X_val, y_val, X_test, y_test)

    # -- Attach baseline for comparison
    metrics["baseline"] = baseline
    metrics["lift_vs_baseline_auprc"] = round(
        metrics["test_auprc"] - baseline["baseline_auprc"], 4
    )

    logger.info(
        "\n" + "="*60 + "\n"
        "  FINAL RESULTS\n"
        f"  Baseline AUPRC : {baseline['baseline_auprc']:.4f}\n"
        f"  Model AUPRC    : {metrics['test_auprc']:.4f}  "
        f"(+{metrics['lift_vs_baseline_auprc']:.4f})\n"
        f"  AUROC          : {metrics['test_auroc']:.4f}\n"
        f"  Precision@thr  : {metrics['test_precision_at_threshold']:.4f}\n"
        f"  Recall@thr     : {metrics['test_recall_at_threshold']:.4f}\n"
        f"  Lift@Top10%    : {metrics['lift_at_top10pct']:.2f}x\n"
        + "="*60
    )

    # -- Persist
    save_model(model)
    save_metrics(metrics)
    save_pr_curve(pr_curve)

    # -- Feature importance
    fi_df = get_feature_importance(model)
    logger.info(f"\nTop features:\n{fi_df.to_string(index=False)}")

    return model, metrics, pr_curve, fi_df


def main() -> None:
    """Entry point for direct script execution."""
    try:
        run_training_pipeline()
        logger.info("✅ Training pipeline complete.")
    except FileNotFoundError as e:
        logger.critical(f"Missing required file: {e}")
        sys.exit(1)
    except ImportError as e:
        logger.critical(f"Missing dependency: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unhandled fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
