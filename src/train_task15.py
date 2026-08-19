"""
train_task15.py — PlaceMux Phase 3, Task 15
===========================================
Initial Model Training and Registration

Design rationale
----------------
This script acts as the "Day 0" setup for the model governance framework.
It trains a baseline Churn Prediction model (v1), evaluates it offline, 
registers it into the `ModelRegistry`, sets it as the active champion, 
and generates the first automated Model Card.

Output
------
- models/churn_model_v1.pkl
- models/registry.json (updated)
- logs/model_card_churn_model_v1.md
- logs/task15.log
"""

import os
import sys
import pickle
import logging
import numpy as np

# Adjust path to import from sibling modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from retraining_pipeline import generate_fresh_data, train_model, evaluate_model, FEATURE_COLS
from model_registry import ModelRegistry
from model_card import ModelCardGenerator

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

def main():
    try:
        logger.info("Starting Task 15 Initial Training...")
        
        # 1. Initialize Registry and Card Generator
        registry = ModelRegistry("models/registry.json")
        card_gen = ModelCardGenerator("logs")
        
        # 2. Generate initial training and validation data (simulating Day 0)
        logger.info("Generating Day 0 baseline data...")
        df_train = generate_fresh_data(n_samples=10000, random_state=42)
        df_val = generate_fresh_data(n_samples=2000, random_state=99)
        
        # 3. Train the model
        logger.info("Training initial model (v1)...")
        model_v1 = train_model(df_train, random_state=42)
        
        # 4. Evaluate the model
        logger.info("Evaluating model v1 offline...")
        metrics = evaluate_model(model_v1, df_val)
        
        # 5. Serialize the artifact
        artifact_path = "models/churn_model_v1.pkl"
        os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
        with open(artifact_path, "wb") as f:
            pickle.dump(model_v1, f)
        logger.info(f"Model saved to {artifact_path}")
        
        # 6. Register model
        lineage = {
            "training_timestamp": df_train.empty is False and "2026-08-19" or "Unknown",
            "n_train_samples": len(df_train),
            "features": FEATURE_COLS,
            "training_script": "train_task15.py"
        }
        
        model_name = "churn_model"
        version = "v1"
        
        registry.register_model(
            model_name=model_name,
            version=version,
            artifact_path=artifact_path,
            metrics=metrics,
            lineage=lineage
        )
        
        # 7. Promote to champion
        registry.promote_to_champion(model_name, version)
        
        # 8. Generate Model Card
        registry_info = registry.get_model(model_name, version)
        card_path = card_gen.generate_card(
            model=model_v1,
            registry_info=registry_info,
            df_val=df_val,
            feature_cols=FEATURE_COLS
        )
        logger.info(f"Task 15 initial setup complete. Model Card at: {card_path}")
        
    except Exception as e:
        logger.critical(f"Unhandled fatal error in training script: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
