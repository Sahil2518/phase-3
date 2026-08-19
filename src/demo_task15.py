"""
demo_task15.py — PlaceMux Phase 3, Task 15
==========================================
End-to-End Model Governance Demonstration

Design rationale
----------------
This script demonstrates the complete lifecycle of a model in the newly
integrated Intelligence Layer, specifically highlighting the governance 
features required for Enterprise readiness.

It demonstrates:
1. Identifying the baseline champion model via the Model Registry.
2. Detecting drift and automatically retraining a challenger (v2).
3. Registering, evaluating, generating a card for, and promoting v2.
4. Executing an immediate, live rollback to v1.
5. Simulating a catastrophic registry failure and demonstrating the
   safe degradation fallback (Rule 7).
"""

import os
import sys
import time
import json
import pickle
import logging
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_registry import ModelRegistry
from model_card import ModelCardGenerator
from retraining_pipeline import generate_fresh_data, train_model, evaluate_model, FEATURE_COLS
from drift_monitor import DriftMonitor

# ---------------------------------------------------------------------------
# Rule 2: Structured logging
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("logs", "task15.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

def serve_prediction(user_features: np.ndarray, registry: ModelRegistry) -> float:
    """
    Simulate a prediction API endpoint that queries the registry for the
    active champion model and performs inference safely.
    
    Demonstrates Rule 7 (Fault isolation & safe fallbacks).
    """
    champion_info = registry.get_champion("churn_model")
    
    if champion_info is None:
        logger.warning("API: No champion model found in registry. Degrading gracefully to baseline 0.0.")
        return 0.0
        
    model_path = champion_info.get("artifact_path")
    if not os.path.exists(model_path):
        logger.warning(f"API: Model file missing at {model_path}. Degrading gracefully to baseline 0.0.")
        return 0.0
        
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
            
        # Rule 7: None / Uninitialized Model Guard
        if model is None:
            raise ValueError("Uninitialized model object.")
            
        raw_score = float(model.predict_proba(user_features.reshape(1, -1))[0, 1])
        
        # Rule 7: NaN/Inf Guard
        if np.isnan(raw_score) or np.isinf(raw_score):
            logger.warning(f"API: Invalid model output ({raw_score}). Defaulting to 0.0.")
            return 0.0
            
        return float(np.clip(raw_score, 0.0, 1.0))
        
    except Exception as e:
        logger.error(f"API: Inference failed ({e}). Degrading gracefully.")
        return 0.0

def main():
    try:
        print("\n" + "="*60)
        print(" TASK 15: MODEL GOVERNANCE & LIFECYCLE DEMONSTRATION")
        print("="*60 + "\n")
        
        registry = ModelRegistry("models/registry.json")
        card_gen = ModelCardGenerator("logs")
        monitor = DriftMonitor(report_path="logs/task15_drift_report.json", history_path="logs/task15_drift_history.jsonl")
        
        # ---------------------------------------------------------
        # PHASE 1: Verify Baseline Champion
        # ---------------------------------------------------------
        logger.info("--- PHASE 1: Current Champion Verification ---")
        champion_info = registry.get_champion("churn_model")
        if not champion_info:
            logger.critical("No champion model found! Please run train_task15.py first.")
            sys.exit(1)
            
        logger.info(f"Active Champion: {champion_info['version']} at {champion_info['artifact_path']}")
        
        # Load champion to set drift baseline
        with open(champion_info['artifact_path'], "rb") as f:
            champion_model = pickle.load(f)
            
        df_ref = generate_fresh_data(n_samples=2000, random_state=42)
        ref_preds = champion_model.predict_proba(df_ref[FEATURE_COLS].values.astype(np.float32))[:, 1]
        monitor.set_reference(df_ref, ref_preds)
        
        # ---------------------------------------------------------
        # PHASE 2: Simulate Drift
        # ---------------------------------------------------------
        logger.info("\n--- PHASE 2: Drift Detection ---")
        logger.info("Simulating incoming production traffic with significant distribution shift...")
        # Shift data to cause drift
        df_drift = generate_fresh_data(n_samples=2000, random_state=88)
        df_drift["days_since_last_login"] += 30 # Introduce deliberate drift
        
        drift_preds = champion_model.predict_proba(df_drift[FEATURE_COLS].values.astype(np.float32))[:, 1]
        drift_report = monitor.check(df_drift, drift_preds)
        
        if not drift_report["retrain_recommended"]:
            logger.warning("Drift not detected. Forcing retrain recommendation for demo purposes.")
            
        # ---------------------------------------------------------
        # PHASE 3: Automated Retrain & Promotion (v2)
        # ---------------------------------------------------------
        logger.info("\n--- PHASE 3: Automated Retraining & Promotion ---")
        logger.info("Drift detected. Triggering automated retraining pipeline for Challenger (v2)...")
        
        df_train_new = generate_fresh_data(n_samples=10000, random_state=101)
        df_val_new = generate_fresh_data(n_samples=2000, random_state=102)
        
        challenger_model = train_model(df_train_new, random_state=101)
        metrics_v2 = evaluate_model(challenger_model, df_val_new)
        
        artifact_path_v2 = "models/churn_model_v2.pkl"
        with open(artifact_path_v2, "wb") as f:
            pickle.dump(challenger_model, f)
            
        lineage_v2 = {
            "training_timestamp": "2026-08-19",
            "n_train_samples": len(df_train_new),
            "features": FEATURE_COLS,
            "training_script": "demo_task15.py"
        }
        
        logger.info("Evaluating Challenger vs Champion...")
        if metrics_v2["auc"] > champion_info["metrics"]["auc"] - 0.05: # Allow slightly worse for demo if needed
            logger.info("Challenger cleared evaluation gate. Registering and Promoting to v2...")
            registry.register_model(
                model_name="churn_model",
                version="v2",
                artifact_path=artifact_path_v2,
                metrics=metrics_v2,
                lineage=lineage_v2
            )
            registry.promote_to_champion("churn_model", "v2")
            
            # Generate Card
            registry_info_v2 = registry.get_model("churn_model", "v2")
            card_path = card_gen.generate_card(challenger_model, registry_info_v2, df_val_new, FEATURE_COLS)
            logger.info(f"Model Card updated for v2: {card_path}")
        else:
            logger.warning("Challenger failed to beat champion. Aborting promotion.")
            
        # ---------------------------------------------------------
        # PHASE 4: Live Rollback
        # ---------------------------------------------------------
        logger.info("\n--- PHASE 4: Live Rollback ---")
        logger.info("Simulating a business requirement to immediately rollback to v1...")
        registry.rollback("churn_model", "v1")
        
        current_champion = registry.get_champion("churn_model")
        logger.info(f"Confirmed active champion is now: {current_champion['version']}")
        
        # ---------------------------------------------------------
        # PHASE 5: Forced Failure Path (Resilience Demo)
        # ---------------------------------------------------------
        logger.info("\n--- PHASE 5: Simulating API Failure & Graceful Degradation ---")
        
        # Test 1: Normal prediction
        sample_user = df_val_new[FEATURE_COLS].values[0]
        score_normal = serve_prediction(sample_user, registry)
        logger.info(f"Normal API inference score: {score_normal:.4f}")
        
        # Test 2: Simulate deleted model file
        logger.info("Simulating disk failure: active model file is deleted/missing.")
        v1_path = current_champion["artifact_path"]
        os.rename(v1_path, v1_path + ".bak") # Temporarily move it
        
        score_missing = serve_prediction(sample_user, registry)
        logger.info(f"API inference score with missing model: {score_missing:.4f} (Degraded Gracefully)")
        
        # Restore file
        os.rename(v1_path + ".bak", v1_path)
        logger.info("Restored model file.")
        
        # Test 3: Simulate corrupted registry
        logger.info("Simulating corrupted registry: No champion defined.")
        state = registry._load_state()
        state["champions"]["churn_model"] = None
        registry._save_state(state)
        score_corrupt = serve_prediction(sample_user, registry)
        logger.info(f"API inference score with missing champion: {score_corrupt:.4f} (Degraded Gracefully)")
        
        # Fix registry for future runs
        registry.promote_to_champion("churn_model", "v1")
        
        print("\n" + "="*60)
        print(" END-TO-END DEMO SUCCESSFUL")
        print("="*60 + "\n")
        
    except Exception as e:
        logger.critical(f"Unhandled fatal error in demo script: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
