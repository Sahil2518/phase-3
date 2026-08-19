"""
demo_task22.py -- PlaceMux Phase 3, Task 22
==========================================
End-to-End Demo: Drift Monitoring + Retraining Pipeline

This script demonstrates the full drift -> detect -> retrain -> promote cycle
in five clear stages:

  Stage A: Establish Baseline
    - Load/generate the training-window data
    - Set the DriftMonitor reference distribution
    - Benchmark the current champion model AUC

  Stage B: Stable Production Window
    - Generate a production window with the same distribution as training
    - Run the DriftMonitor -- expect STABLE result

  Stage C: Drifted Production Window
    - Generate a drifted production window (shifted feature distributions)
    - Run the DriftMonitor -- expect DRIFT_DETECTED + retrain_recommended=True

  Stage D: Automated Retraining
    - Trigger the RetrainingPipeline with reason='drift_detected'
    - Print champion/challenger AUC comparison
    - Confirm promotion decision

  Stage E: History Summary
    - Summarise drift history and output files

Usage:
    python -m src.demo_task22
    # or via run_task22.bat
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Rule 2: Structured logging (ASCII-safe for Windows CP-1252 terminals)
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)

# Force UTF-8 on the file handler; leave stream handler to system default
_file_handler   = logging.FileHandler(os.path.join("logs", "task22.log"), encoding="utf-8")
_stream_handler = logging.StreamHandler(sys.stdout)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[_file_handler, _stream_handler],
)
logger = logging.getLogger(__name__)

# Local imports
from src.drift_monitor import DriftMonitor
from src.retraining_pipeline import (
    RetrainingPipeline,
    generate_fresh_data,
    train_model,
    evaluate_model,
    get_current_champion_path,
    FEATURE_COLS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_section(title: str) -> None:
    """Print a visually distinct section header (ASCII-safe)."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def _make_stable_window(rng: np.random.Generator, n: int = 800) -> pd.DataFrame:
    """
    Build a stable production window that matches the reference distribution.

    The reference was generated with `generate_fresh_data(n_samples=8000, seed=42)`.
    This function produces data from the same distributions so PSI stays near 0.

    Parameters
    ----------
    rng : np.random.Generator
        Shared RNG (seeded externally for reproducibility).
    n : int
        Number of samples to generate.

    Returns
    -------
    pd.DataFrame
        Feature DataFrame with the same column set as FEATURE_COLS.
    """
    s14 = rng.integers(0, 18, n).astype(float)
    s30 = s14 + rng.integers(0, 20, n).astype(float)  # same formula as generate_fresh_data
    return pd.DataFrame({
        "days_since_last_login":  rng.integers(0, 60, n).astype(float),
        "sessions_last_14d":      s14,
        "sessions_last_30d":      s30,
        "apply_rate_7d":          rng.uniform(0, 0.45, n),
        "jobs_viewed_lifetime":   rng.integers(0, 200, n).astype(float),
        "recruiter_contacts":     rng.integers(0, 15, n).astype(float),
        "days_since_first_login": rng.integers(30, 365, n).astype(float),
        "profile_completeness":   rng.uniform(20, 100, n),
        "is_profile_verified":    rng.integers(0, 2, n).astype(float),
    })


def _make_drifted_window(rng: np.random.Generator, n: int = 800) -> pd.DataFrame:
    """
    Build a heavily drifted production window.

    Distributions are intentionally shifted to simulate:
    - Users becoming significantly less active (longer absence, fewer sessions)
    - Near-zero application activity
    - More incomplete profiles

    Parameters
    ----------
    rng : np.random.Generator
    n : int

    Returns
    -------
    pd.DataFrame
    """
    return pd.DataFrame({
        "days_since_last_login":  rng.integers(40, 100, n).astype(float),  # SHIFT: long absence
        "sessions_last_14d":      rng.integers(0, 3, n).astype(float),     # SHIFT: near-zero
        "sessions_last_30d":      rng.integers(0, 5, n).astype(float),     # SHIFT: near-zero
        "apply_rate_7d":          rng.uniform(0, 0.03, n),                 # SHIFT: almost no applies
        "jobs_viewed_lifetime":   rng.integers(0, 25, n).astype(float),    # SHIFT: low engagement
        "recruiter_contacts":     rng.integers(0, 3, n).astype(float),
        "days_since_first_login": rng.integers(30, 365, n).astype(float),
        "profile_completeness":   rng.uniform(10, 50, n),                  # SHIFT: incomplete
        "is_profile_verified":    rng.integers(0, 2, n).astype(float),
    })


def load_or_train_baseline_model(random_state: int = 42):
    """
    Load the existing champion model, or train a new v1 if none exists.

    Parameters
    ----------
    random_state : int

    Returns
    -------
    tuple of (model, str)
        (loaded/trained model, champion file path)
    """
    import pickle

    champ_path = get_current_champion_path()
    if champ_path and os.path.exists(champ_path):
        logger.info(f"Loading existing champion: {champ_path}")
        with open(champ_path, "rb") as f:
            return pickle.load(f), champ_path

    logger.info("No champion found -- training v1 baseline model.")
    df = generate_fresh_data(n_samples=8000, random_state=random_state)
    model = train_model(df, random_state=random_state)

    champ_path = "models/churn_model_v1.pkl"
    with open(champ_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Baseline champion saved: {champ_path}")
    return model, champ_path


# ---------------------------------------------------------------------------
# Main Demo
# ---------------------------------------------------------------------------

def run_demo() -> None:
    """
    Execute the full Task 22 drift + retraining demo.

    All print output uses ASCII-only markers ([OK], [WARN], etc.)
    to avoid Windows CP-1252 encoding failures.
    """
    rng = np.random.default_rng(42)

    # =========================================================================
    # STAGE A: Establish Baseline
    # =========================================================================
    print_section("Stage A: Establishing Baseline (Model + Reference Distribution)")

    champion, champ_path = load_or_train_baseline_model(random_state=42)
    print(f"  Champion model: {champ_path}")

    # Generate training-window reference data (same params as generate_fresh_data)
    ref_df = generate_fresh_data(n_samples=8000, random_state=42)
    X_ref  = ref_df[FEATURE_COLS].values.astype("float32")
    ref_preds = champion.predict_proba(X_ref)[:, 1]

    # Initialise DriftMonitor
    monitor = DriftMonitor(
        drift_threshold_psi=0.25,
        drift_threshold_jsd=0.10,
    )
    monitor.set_reference(
        features_df=ref_df[FEATURE_COLS],
        predictions=ref_preds,
    )

    # Baseline evaluation
    eval_df = generate_fresh_data(n_samples=2000, random_state=99)
    baseline_metrics = evaluate_model(champion, eval_df)
    print(f"\n  Baseline Champion Metrics:")
    print(f"    AUC : {baseline_metrics['auc']:.4f}")
    print(f"    F1  : {baseline_metrics['f1']:.4f}")

    # =========================================================================
    # STAGE B: Stable Production Window
    # =========================================================================
    print_section("Stage B: Stable Production Window")

    stable_df   = _make_stable_window(rng, n=800)
    X_stable    = stable_df[FEATURE_COLS].values.astype("float32")
    stable_preds = champion.predict_proba(X_stable)[:, 1]

    report_stable = monitor.check(stable_df[FEATURE_COLS], stable_preds)
    status_s = report_stable["overall_status"]
    print(f"\n  [{'OK  ' if status_s == 'STABLE' else 'WARN'}] Drift Status      : {status_s}")
    print(f"  [{'OK  ' if not report_stable['retrain_recommended'] else 'WARN'}] Retrain Needed    : {report_stable['retrain_recommended']}")
    print(f"  Features Drifted  : {report_stable['n_features_drifted']}/{report_stable['n_features_checked']}")
    print(f"  Concept Drift JSD : {report_stable['concept_drift']['jsd']:.4f}")

    print(f"\n  Feature PSI (Stable Window):")
    print(f"    {'Feature':<30} {'PSI':>8}  Severity")
    print(f"    {'-'*52}")
    for feat, res in report_stable["feature_drift"].items():
        print(f"    {feat:<30} {res['psi']:>8.5f}  {res['severity']}")

    # =========================================================================
    # STAGE C: Drifted Production Window
    # =========================================================================
    print_section("Stage C: Drifted Production Window (Simulating Data Drift)")

    drifted_df    = _make_drifted_window(rng, n=800)
    X_drifted     = drifted_df[FEATURE_COLS].values.astype("float32")
    drifted_preds = champion.predict_proba(X_drifted)[:, 1]

    report_drifted = monitor.check(drifted_df[FEATURE_COLS], drifted_preds)
    status_d = report_drifted["overall_status"]
    print(f"\n  [{'WARN' if report_drifted['retrain_recommended'] else 'OK  '}] Drift Status      : {status_d}")
    print(f"  [{'WARN' if report_drifted['retrain_recommended'] else 'OK  '}] Retrain Needed    : {report_drifted['retrain_recommended']}")
    print(f"  Features Drifted  : {report_drifted['n_features_drifted']}/{report_drifted['n_features_checked']}")
    print(f"  Concept Drift JSD : {report_drifted['concept_drift']['jsd']:.4f}")
    print(f"  Pred Mean (Ref)   : {report_drifted['concept_drift']['prediction_mean_ref']:.4f}")
    print(f"  Pred Mean (Curr)  : {report_drifted['concept_drift']['prediction_mean_current']:.4f}")
    print(f"  Pred Mean Shift   : {report_drifted['concept_drift']['prediction_mean_shift']:+.4f}")

    print(f"\n  Feature PSI (Drifted Window):")
    print(f"    {'Feature':<30} {'PSI':>8}  {'Severity':<8}  Drifted?")
    print(f"    {'-'*62}")
    for feat, res in report_drifted["feature_drift"].items():
        marker = "[DRIFT]" if res["drifted"] else "[OK]   "
        print(f"    {feat:<30} {res['psi']:>8.5f}  {res['severity']:<8}  {marker}")

    if report_drifted["drifted_features"]:
        print(f"\n  Drifted features: {', '.join(report_drifted['drifted_features'])}")

    # =========================================================================
    # STAGE D: Automated Retraining
    # =========================================================================
    print_section("Stage D: Automated Retraining (Triggered by Drift)")

    retrain_report = {}
    if report_drifted["retrain_recommended"]:
        print("\n  Drift confirmed -- triggering RetrainingPipeline...")
        pipeline = RetrainingPipeline(
            improvement_threshold=0.005,
            n_samples=8000,
            val_fraction=0.2,
            random_state=42,
        )
        retrain_report = pipeline.run(trigger_reason="drift_detected")

        print(f"\n  Retrain Status         : {retrain_report['status']}")
        print(f"  Champion AUC (before)  : {retrain_report['champion']['auc']:.4f}")
        print(f"  Challenger AUC         : {retrain_report['challenger']['auc']:.4f}")
        print(f"  Challenger F1          : {retrain_report['challenger']['f1']:.4f}")
        print(f"  AUC Improvement        : {retrain_report['improvement']:+.4f}")
        promoted_str = "YES [PROMOTED]" if retrain_report.get("promoted") else "NO  [RETAINED champion]"
        print(f"  Promoted?              : {promoted_str}")
        if retrain_report.get("promoted"):
            print(f"  New Champion Path      : {retrain_report['challenger']['path']}")
    else:
        print("  No retrain needed -- skipping pipeline.")

    # =========================================================================
    # STAGE E: Summary
    # =========================================================================
    print_section("Stage E: Drift & Retrain History Summary")

    drift_summary = monitor.summarise_history()
    print(f"\n  Drift Monitor History:")
    print(f"    Total Checks   : {drift_summary.get('total_checks', 0)}")
    print(f"    Drift Events   : {drift_summary.get('n_drift_events', 0)}")
    print(f"    Drift Rate     : {drift_summary.get('drift_rate', 0):.0%}")
    print(f"    Last Status    : {drift_summary.get('last_status', 'N/A')}")

    print(f"\n  Output Files:")
    for path in [
        "logs/drift_report.json",
        "logs/drift_history.jsonl",
        "logs/retrain_report.json",
        "logs/retrain_history.jsonl",
        "logs/task22.log",
    ]:
        exists = "[OK]" if os.path.exists(path) else "[MISS]"
        print(f"    {exists} {path}")

    print("\n" + "=" * 70)
    print("  Task 22 Demo Complete.")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Main entry point with top-level error handling (Rule 2)."""
    try:
        run_demo()
    except FileNotFoundError as e:
        logger.critical(f"Missing required file: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unhandled fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
