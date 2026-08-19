"""
model_registry.py — PlaceMux Phase 3, Task 15
=============================================
Central Model Registry for governance, versioning, and lineage tracking.

Design rationale
----------------
This module provides a robust backend for tracking models from training to 
production. It stores metadata such as offline metrics, data lineage, and 
the absolute path to the pickled artifacts in a central JSON store.

It supports:
1. Registration of new model versions.
2. Promotion of a specific version to 'champion'.
3. Safe rollback of the champion alias to a previous version in case of failure.

Output
------
- models/registry.json : The single source of truth for model metadata.
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Optional, Any

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

class ModelRegistry:
    """
    Manages the lifecycle, versioning, and state of machine learning models.
    
    Attributes
    ----------
    registry_path : str
        Path to the JSON file where the registry state is persisted.
    """
    
    def __init__(self, registry_path: str = "models/registry.json") -> None:
        """
        Initialise the registry store.
        
        Parameters
        ----------
        registry_path : str
            Filepath to the JSON registry.
        """
        self.registry_path = registry_path
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        self._initialize_registry()

    def _initialize_registry(self) -> None:
        """Create the registry JSON file if it doesn't exist."""
        if not os.path.exists(self.registry_path):
            state = {"models": {}, "champions": {}}
            self._save_state(state)
            logger.info(f"Initialized new model registry at {self.registry_path}")

    def _load_state(self) -> dict:
        """Load the current registry state from disk."""
        try:
            with open(self.registry_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load registry from {self.registry_path}: {e}")
            return {"models": {}, "champions": {}}

    def _save_state(self, state: dict) -> None:
        """Persist the registry state to disk."""
        try:
            with open(self.registry_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save registry to {self.registry_path}: {e}")

    def register_model(
        self, 
        model_name: str, 
        version: str, 
        artifact_path: str, 
        metrics: Dict[str, float], 
        lineage: Dict[str, Any]
    ) -> None:
        """
        Register a new model version with its metadata.

        Parameters
        ----------
        model_name : str
            The logical name of the model (e.g., 'churn_prediction').
        version : str
            The version identifier (e.g., 'v1', 'v2').
        artifact_path : str
            Path to the serialized model file.
        metrics : dict
            Offline evaluation metrics (e.g., {'auc': 0.85, 'f1': 0.7}).
        lineage : dict
            Information about training data, features, and timestamp.
        """
        state = self._load_state()
        
        if model_name not in state["models"]:
            state["models"][model_name] = {}
            
        record = {
            "model_name": model_name,
            "version": version,
            "artifact_path": artifact_path,
            "metrics": metrics,
            "lineage": lineage,
            "registered_at": datetime.utcnow().isoformat()
        }
        
        state["models"][model_name][version] = record
        self._save_state(state)
        logger.info(f"Registered {model_name} version {version} at {artifact_path}")

    def promote_to_champion(self, model_name: str, version: str) -> None:
        """
        Promote a registered model version to be the active champion.

        Parameters
        ----------
        model_name : str
            The logical name of the model.
        version : str
            The version to promote.
        """
        state = self._load_state()
        if model_name not in state["models"] or version not in state["models"][model_name]:
            raise ValueError(f"Cannot promote: Model {model_name} version {version} not found in registry.")
            
        state["champions"][model_name] = version
        self._save_state(state)
        logger.info(f"Promoted {model_name} version {version} to CHAMPION.")

    def rollback(self, model_name: str, target_version: str) -> None:
        """
        Safely roll back the champion to a previous version.

        Parameters
        ----------
        model_name : str
            The logical name of the model.
        target_version : str
            The version to roll back to.
        """
        state = self._load_state()
        if model_name not in state["models"] or target_version not in state["models"][model_name]:
            raise ValueError(f"Rollback failed: Target version {target_version} not found for {model_name}.")
            
        current_champion = state["champions"].get(model_name)
        if current_champion == target_version:
            logger.warning(f"{model_name} is already at champion version {target_version}. Rollback is a no-op.")
            return

        state["champions"][model_name] = target_version
        self._save_state(state)
        logger.warning(f"ROLLED BACK {model_name} from {current_champion} to {target_version}.")

    def get_champion(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the metadata and path for the current champion model.

        Parameters
        ----------
        model_name : str
            The logical name of the model.

        Returns
        -------
        dict or None
            The registry record of the champion, or None if no champion exists.
        """
        state = self._load_state()
        version = state["champions"].get(model_name)
        if not version:
            logger.warning(f"No champion set for model {model_name}.")
            return None
            
        return state["models"].get(model_name, {}).get(version)

    def get_model(self, model_name: str, version: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve metadata for a specific model version.

        Parameters
        ----------
        model_name : str
            The logical name of the model.
        version : str
            The version identifier.

        Returns
        -------
        dict or None
            The registry record, or None if not found.
        """
        state = self._load_state()
        return state.get("models", {}).get(model_name, {}).get(version)
