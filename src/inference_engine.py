import pandas as pd
import numpy as np
import logging
import math

# Rule 2: Structured Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

class UnoptimizedInferenceEngine:
    """
    Simulates a slow, unoptimized ML scoring service using Pandas iteration.
    This mimics real-world bottlenecks where Python for-loops dominate execution time.
    """
    
    def __init__(self):
        self.model_loaded = True
        
    def predict(self, df: pd.DataFrame) -> list:
        """
        Calculates scores using unoptimized row-by-row DataFrame operations.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input features containing 'skill_score' and 'experience_years'.
            
        Returns
        -------
        scores : list
            A list of prediction scores between 0 and 1.
        """
        # Rule 7: Guard for uninitialized model
        if not self.model_loaded:
            raise ValueError("Cannot predict: model is uninitialized or None.")
            
        # Rule 7: Empty input guard
        if df is None or df.empty:
            logger.warning("Empty input DataFrame provided. Returning empty response.")
            return []
            
        scores = []
        # Bottleneck: Iterating over DataFrame rows
        for index, row in df.iterrows():
            try:
                # Simulated complex feature engineering / scoring math
                skill = row['skill_score']
                exp = row['experience_years']
                
                # Math heavy operation to exaggerate the bottleneck
                base_score = math.pow(skill, 1.2) + math.log1p(exp)
                raw_score = base_score / 10.0
                
                # Rule 7: NaN / Infinity Output Guard
                if math.isnan(raw_score) or math.isinf(raw_score):
                    logger.warning(f"Invalid model output ({raw_score}). Defaulting to 0.0.")
                    score = 0.0
                else:
                    # Clip score
                    score = max(0.0, min(1.0, raw_score))
                    
                scores.append(score)
                
            except Exception as e:
                # Rule 7: Fault Isolation (per-item in batch)
                logger.error(f"Failed to score row {index}: {e}")
                scores.append(0.0)
                
        return scores


class OptimizedInferenceEngine:
    """
    Simulates an optimized ML scoring service using vectorized NumPy operations.
    Maintains exact mathematical equivalence with UnoptimizedInferenceEngine.
    """
    
    def __init__(self):
        self.model_loaded = True
        
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculates scores using fast, vectorized NumPy operations.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input features containing 'skill_score' and 'experience_years'.
            
        Returns
        -------
        scores : np.ndarray
            An array of prediction scores between 0 and 1.
        """
        if not self.model_loaded:
            raise ValueError("Cannot predict: model is uninitialized or None.")
            
        if df is None or df.empty:
            logger.warning("Empty input DataFrame provided. Returning empty response.")
            return np.array([])
            
        try:
            skill = df['skill_score'].values
            exp = df['experience_years'].values
            
            # Vectorized operations (identically equivalent math)
            base_score = np.power(skill, 1.2) + np.log1p(exp)
            raw_score = base_score / 10.0
            
            # Vectorized clip
            scores = np.clip(raw_score, 0.0, 1.0)
            
            # Rule 7: NaN / Infinity Guard
            invalid_mask = np.isnan(scores) | np.isinf(scores)
            if np.any(invalid_mask):
                logger.warning(f"Found {np.sum(invalid_mask)} invalid model outputs. Defaulting to 0.0.")
                scores[invalid_mask] = 0.0
                
            return scores
            
        except Exception as e:
            logger.error(f"Vectorized prediction failed: {e}", exc_info=True)
            # Fallback to zero arrays
            return np.zeros(len(df))
