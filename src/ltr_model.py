"""
ltr_model.py -- PlaceMux Phase 3, Task 11
=========================================
Learning-to-Rank (LTR) model with Inverse Propensity Scoring (IPS).

Trains a pairwise ranking model (LambdaMART) using LightGBM.
Corrects for position bias by weighting training samples by the
inverse of their estimated examination probability.

Also provides the heuristic baseline for comparison.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Dict, List

try:
    import lightgbm as lgb
except ImportError:
    logging.warning("lightgbm not found. 'pip install lightgbm' is required for LTR.")
    lgb = None

logger = logging.getLogger(__name__)


class IPSRanker:
    """
    LightGBM Ranker trained with Inverse Propensity Scoring.
    """

    def __init__(self, features: List[str]):
        self.features = features
        self.model = None
        self.propensity_scores: Dict[int, float] = {}

    def _estimate_propensities(self, df: pd.DataFrame) -> None:
        """
        Estimate examination probability by position relative to position 1.
        Assuming relevance is distributed randomly across positions in logged data,
        P(click | pos=k) is proportional to P(exam | pos=k).
        """
        clicks_by_pos = df.groupby("position")["click"].mean()
        if 1 not in clicks_by_pos:
            raise ValueError("Position 1 is required to anchor propensities.")

        base_ctr = clicks_by_pos[1]
        
        for pos, ctr in clicks_by_pos.items():
            # Avoid division by zero or negative props
            prop = max(0.01, ctr / base_ctr)
            self.propensity_scores[pos] = min(1.0, prop)
            
        logger.info(f"Estimated Propensity Scores: {self.propensity_scores}")

    def train(self, df_train: pd.DataFrame, df_val: pd.DataFrame = None) -> None:
        """
        Train the LGBMRanker using IPS weights.
        """
        if lgb is None:
            raise ImportError("Cannot train LTR model without lightgbm.")

        # Estimate propensities on train set
        self._estimate_propensities(df_train)

        # Sort by query_id so LightGBM can form groups
        df_train = df_train.sort_values("query_id").reset_index(drop=True)
        
        # Calculate IPS weights: 1.0 / P(exam | pos)
        # Cap max weight to avoid exploding gradients on deep positions
        max_weight = 10.0
        weights = df_train["position"].map(
            lambda p: min(max_weight, 1.0 / self.propensity_scores.get(p, 0.01))
        ).values

        X_train = df_train[self.features]
        y_train = df_train["click"]
        q_train = df_train.groupby("query_id").size().values

        eval_set = None
        eval_group = None
        
        if df_val is not None:
            df_val = df_val.sort_values("query_id").reset_index(drop=True)
            X_val = df_val[self.features]
            y_val = df_val["click"]
            q_val = df_val.groupby("query_id").size().values
            eval_set = [(X_val, y_val)]
            eval_group = [q_val]

        self.model = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            importance_type="gain",
        )

        logger.info("Training LGBMRanker with IPS weights...")
        self.model.fit(
            X_train,
            y_train,
            group=q_train,
            sample_weight=weights,
            eval_set=eval_set,
            eval_group=eval_group,
            eval_at=[5, 10],
            callbacks=[lgb.early_stopping(stopping_rounds=10)] if eval_set else None
        )
        logger.info("Training complete.")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict ranking scores for the given candidates.
        """
        if self.model is None:
            raise RuntimeError("Model is not trained.")
        X = df[self.features]
        return self.model.predict(X)


class HeuristicRanker:
    """
    The baseline ranking heuristic from Phase 2 / early Phase 3.
    Uses a simple weighted sum of features.
    """
    def __init__(self, features: List[str]):
        self.features = features

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Score using the static heuristic formula.
        """
        # Ensure missing features don't crash
        m = df["match_score"] if "match_score" in df else 0.5
        l = df["loc_match"] if "loc_match" in df else 0.5
        r = df["recency_days"] if "recency_days" in df else 15.0
        s = df["seniority_delta"] if "seniority_delta" in df else 0.0

        scores = (
            0.4 * m
            + 0.3 * l
            - 0.1 * (r / 30.0)
            - 0.2 * np.abs(s)
        )
        return scores.values
