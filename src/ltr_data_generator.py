"""
ltr_data_generator.py -- PlaceMux Phase 3, Task 11
=================================================
Generates simulated impression logs for Learning-to-Rank (LTR).

Simulates search/recommendation queries where a set of candidates (jobs)
are presented to a user at specific rank positions. Injects strong
position bias: users are much more likely to examine and click top
positions regardless of true relevance.

This provides the synthetic data needed to train and evaluate the
LTR model and its inverse propensity scoring (IPS) bias correction.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Tuple

logger = logging.getLogger(__name__)


class LTRDataGenerator:
    """
    Generates synthetic query-candidate impressions with position bias.
    """

    def __init__(self, random_state: int = 42):
        self.rng = np.random.default_rng(random_state)
        # Position bias curve: probability of examining a given position (1-indexed)
        # e.g., pos 1 has 90% exam rate, pos 2 has 60%, pos 5 has 20%.
        self.position_exam_prob = {
            1: 0.90,
            2: 0.60,
            3: 0.40,
            4: 0.30,
            5: 0.20,
        }
        # Fallback for positions > 5
        self.default_exam_prob = 0.10

    def get_exam_prob(self, position: int) -> float:
        """Get the true probability that a user examines this position."""
        return self.position_exam_prob.get(position, self.default_exam_prob)

    def generate_dataset(
        self,
        num_queries: int = 1000,
        candidates_per_query: int = 10,
    ) -> pd.DataFrame:
        """
        Generate a synthetic dataset of impressions.

        Features generated:
        - match_score: base relevance [0, 1]
        - loc_match: binary 0/1
        - recency_days: age of listing (lower is better)
        - seniority_delta: match of experience level (0 is perfect)

        Labels:
        - true_relevance: hidden continuous value
        - click: binary outcome (requires both relevance and examination)

        Parameters
        ----------
        num_queries : int
            Number of distinct queries (e.g. users opening the app).
        candidates_per_query : int
            Number of candidates shown per query.

        Returns
        -------
        pd.DataFrame
            The generated dataset.
        """
        rows = []

        for q_idx in range(num_queries):
            query_id = f"q_{q_idx:05d}"

            for c_idx in range(candidates_per_query):
                # 1-indexed position
                position = c_idx + 1

                # Generate features
                match_score = self.rng.uniform(0.1, 0.9)
                loc_match = self.rng.binomial(1, 0.5)
                recency_days = self.rng.integers(1, 30)
                seniority_delta = self.rng.normal(0, 1)

                # Compute true underlying relevance.
                # We intentionally make it so recency is the single most important factor.
                # The heuristic heavily weights `match_score`, so it will be systematically wrong.
                # The LTR model will learn the true importance from clicks and beat the heuristic.
                relevance = (
                    0.8 * (1.0 - (recency_days / 30.0))
                    + 0.1 * match_score
                    + 0.1 * loc_match
                    - 0.05 * abs(seniority_delta)
                )
                # Normalize roughly to [0, 1] for click probability
                relevance_prob = 1.0 / (1.0 + np.exp(-10 * (relevance - 0.2)))

                # Inject position bias: must be examined to be clicked
                exam_prob = self.get_exam_prob(position)
                is_examined = self.rng.binomial(1, exam_prob)

                # Click happens if examined AND relevant
                click = 1 if (is_examined and self.rng.binomial(1, relevance_prob)) else 0

                rows.append({
                    "query_id": query_id,
                    "candidate_id": f"c_{q_idx:05d}_{position:02d}",
                    "position": position,
                    "match_score": match_score,
                    "loc_match": float(loc_match),
                    "recency_days": float(recency_days),
                    "seniority_delta": seniority_delta,
                    "true_relevance": relevance,  # purely for oracle evaluation
                    "click": click,
                })

        df = pd.DataFrame(rows)
        return df

    def split_train_test(
        self, df: pd.DataFrame, test_ratio: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split by query_id to ensure no leakage across queries.
        """
        unique_queries = df["query_id"].unique().tolist()
        self.rng.shuffle(unique_queries)

        split_idx = int(len(unique_queries) * (1 - test_ratio))
        train_queries = set(unique_queries[:split_idx])

        train_df = df[df["query_id"].isin(train_queries)].copy()
        test_df = df[~df["query_id"].isin(train_queries)].copy()

        return train_df, test_df
