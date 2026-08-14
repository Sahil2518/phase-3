"""
ltr_evaluator.py -- PlaceMux Phase 3, Task 11
=============================================
Offline evaluation metrics for Learning-to-Rank models.

Computes Normalized Discounted Cumulative Gain (nDCG@k) and
Mean Average Precision (MAP) by comparing model-predicted ranks
against ground-truth relevance (or binary clicks).
"""

import math
import numpy as np
import pandas as pd
from typing import Dict


def dcg_at_k(relevances: np.ndarray, k: int) -> float:
    """
    Compute Discounted Cumulative Gain at k.
    """
    relevances = np.asarray(relevances, dtype=float)[:k]
    if relevances.size == 0:
        return 0.0
    
    # Gain = 2^rel - 1
    gains = np.power(2, relevances) - 1.0
    # Discounts = log2(rank + 1)
    discounts = np.log2(np.arange(2, relevances.size + 2))
    
    return np.sum(gains / discounts)


def ndcg_at_k(relevances: np.ndarray, k: int) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at k.
    """
    dcg = dcg_at_k(relevances, k)
    if dcg == 0.0:
        return 0.0
        
    ideal_relevances = np.sort(relevances)[::-1]
    idcg = dcg_at_k(ideal_relevances, k)
    
    if idcg == 0.0:
        return 0.0
        
    return dcg / idcg


def average_precision(relevances: np.ndarray, k: int, threshold: float = 0.5) -> float:
    """
    Compute Average Precision at k for binary relevance.
    Treats relevances > threshold as relevant (1), else 0.
    """
    rel = (np.asarray(relevances, dtype=float)[:k] >= threshold).astype(int)
    if rel.sum() == 0:
        return 0.0

    precisions = [
        np.sum(rel[:i+1]) / (i + 1)
        for i in range(len(rel)) if rel[i]
    ]
    return np.sum(precisions) / len(precisions)


class LTREvaluator:
    """
    Evaluates rankers on a test dataframe.
    """

    def __init__(self, k: int = 10, relevance_col: str = "true_relevance"):
        self.k = k
        self.relevance_col = relevance_col

    def evaluate(self, df: pd.DataFrame, score_col: str) -> Dict[str, float]:
        """
        Evaluate a single score column's ranking performance.
        """
        ndcgs = []
        aps = []

        # We determine a binary threshold for MAP based on the median relevance
        rel_threshold = df[self.relevance_col].median()

        for qid, group in df.groupby("query_id"):
            # Sort by predicted score descending
            sorted_group = group.sort_values(score_col, ascending=False)
            relevances = sorted_group[self.relevance_col].values
            
            ndcgs.append(ndcg_at_k(relevances, self.k))
            aps.append(average_precision(relevances, self.k, threshold=rel_threshold))

        return {
            f"ndcg@{self.k}": np.mean(ndcgs),
            f"map@{self.k}": np.mean(aps),
        }

    def compare_models(
        self, df: pd.DataFrame, model_scores: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """
        Compare multiple models. 
        model_scores = {"LTR": array, "Heuristic": array}
        """
        eval_df = df.copy()
        results = {}

        for name, scores in model_scores.items():
            eval_df[name] = scores
            results[name] = self.evaluate(eval_df, name)

        # Format as DataFrame
        res_df = pd.DataFrame(results).T
        return res_df
