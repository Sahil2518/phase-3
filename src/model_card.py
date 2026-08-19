"""
model_card.py — PlaceMux Phase 3, Task 15
=========================================
Automated Model Transparency Card Generator.

Design rationale
----------------
Model cards are essential for governance, explaining how a model works,
its data lineage, performance metrics, and identified limitations (fairness).
This generator produces a Markdown artifact (`logs/model_card_<version>.md`)
that is easily readable by compliance and DevOps teams.

It automatically segments the evaluation set by `is_profile_verified` 
to report any fairness gap (disparate performance).

Output
------
- logs/model_card_{version}.md
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any

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

class ModelCardGenerator:
    """
    Generates Markdown model cards containing metrics, lineage, and fairness analysis.
    """
    
    def __init__(self, output_dir: str = "logs") -> None:
        """
        Initialise generator.
        
        Parameters
        ----------
        output_dir : str
            Directory to save model cards.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _calculate_fairness(self, model, df_val: pd.DataFrame, feature_cols: list) -> Dict[str, dict]:
        """
        Evaluate performance disparity across the 'is_profile_verified' slice.
        """
        if "is_profile_verified" not in df_val.columns:
            logger.warning("Fairness column 'is_profile_verified' not found in validation set.")
            return {}
            
        results = {}
        for slice_val, slice_name in [(1.0, "Verified"), (0.0, "Unverified")]:
            sub_df = df_val[df_val["is_profile_verified"] == slice_val]
            if sub_df.empty:
                continue
                
            X = sub_df[feature_cols].values.astype(np.float32)
            y = sub_df["churned"].values.astype(int)
            
            try:
                probs = model.predict_proba(X)[:, 1]
                preds = (np.clip(probs, 0.0, 1.0) >= 0.5).astype(int)
                
                tp = ((preds == 1) & (y == 1)).sum()
                fp = ((preds == 1) & (y == 0)).sum()
                fn = ((preds == 0) & (y == 1)).sum()
                tn = ((preds == 0) & (y == 0)).sum()
                
                tpr = tp / max(tp + fn, 1)  # Recall
                fpr = fp / max(fp + tn, 1)  # False Positive Rate
                
                results[slice_name] = {
                    "N": len(y),
                    "TPR": round(tpr, 3),
                    "FPR": round(fpr, 3)
                }
            except Exception as e:
                logger.error(f"Failed to calculate fairness for {slice_name}: {e}")
                
        return results

    def generate_card(
        self, 
        model, 
        registry_info: Dict[str, Any], 
        df_val: pd.DataFrame, 
        feature_cols: list
    ) -> str:
        """
        Generate and save the model card.

        Parameters
        ----------
        model : Any
            The loaded machine learning model.
        registry_info : dict
            Metadata retrieved from ModelRegistry.
        df_val : pd.DataFrame
            Validation dataset for fairness analysis.
        feature_cols : list
            List of feature names used by the model.

        Returns
        -------
        str
            The file path to the generated model card.
        """
        model_name = registry_info.get("model_name", "unknown_model")
        version = registry_info.get("version", "v0")
        metrics = registry_info.get("metrics", {})
        lineage = registry_info.get("lineage", {})
        
        # Rule 7: Uninitialized model guard
        if model is None:
            raise ValueError("Cannot generate model card: model is uninitialized or None.")
            
        fairness_results = self._calculate_fairness(model, df_val, feature_cols)
        
        card_content = f"""# Model Card: {model_name}
**Version:** {version}  
**Generated At:** {registry_info.get("registered_at", "Unknown")}  

## 1. Overview
This model is deployed within the PlaceMux Intelligence Layer to predict `{model_name}`. It is governed by the automated retraining pipeline and registered in the central Model Registry.

## 2. Performance Metrics (Offline Validation)
- **ROC-AUC:** {metrics.get('auc', 'N/A')}
- **F1 Score:** {metrics.get('f1', 'N/A')}
- **Validation Samples:** {metrics.get('n_val', 'N/A')}

## 3. Data Lineage & Features
- **Training Timestamp:** {lineage.get('training_timestamp', 'N/A')}
- **Training Samples:** {lineage.get('n_train_samples', 'N/A')}
- **Features Used:** 
{chr(10).join([f"  - `{f}`" for f in feature_cols])}

## 4. Fairness & Bias Analysis
We evaluate model performance parity across sensitive slices. Here, we analyze the disparity between Verified and Unverified user profiles.

| Slice | Sample Size (N) | True Positive Rate (Recall) | False Positive Rate |
|-------|-----------------|-----------------------------|---------------------|
"""
        for slice_name, stats in fairness_results.items():
            card_content += f"| {slice_name} | {stats['N']} | {stats['TPR']} | {stats['FPR']} |\n"

        card_content += """
**Analysis:** If the False Positive Rate (FPR) is significantly higher for one group, the model may be unfairly penalizing them. This gap should be monitored over time.

## 5. Limitations & Rollback Path
- **Known Limitations:** The model assumes user behavior patterns are relatively stable. Sudden macroscopic changes (e.g., website outages) are not captured by these features and may lead to temporary performance degradation.
- **Rollback Path:** If drift or degradation is detected, the `ModelRegistry` permits an immediate live rollback to the previous champion version via `registry.rollback(model_name, previous_version)`.
"""

        output_path = os.path.join(self.output_dir, f"model_card_{model_name}_{version}.md")
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(card_content)
            logger.info(f"Model card generated successfully at {output_path}")
        except Exception as e:
            logger.error(f"Failed to write model card: {e}")
            
        return output_path
