import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any
from src.two_sided_recommender import CatalogManager

logger = logging.getLogger(__name__)

class RecommenderEvaluator:
    def __init__(self, catalog: CatalogManager, k: int = 10):
        self.catalog = catalog
        self.k = k
        self.test_candidates = []
        self._generate_ground_truth()
        
    def _generate_ground_truth(self):
        logger.info("Generating ground truth held-out data...")
        rng = np.random.default_rng(999)
        # Select 200 candidates for evaluation
        cands = self.catalog.candidates_df.sample(200, random_state=999).index
        
        from src.two_sided_recommender import TwoSidedRecommender
        oracle = TwoSidedRecommender(self.catalog)
        
        self.test_candidates = []
        for cand_id in cands:
            # We use oracle to find the actually good jobs
            all_recs = oracle.recommend_jobs_for_candidate(cand_id, k=50)
            # Add some noise to simulate real users
            relevant = set()
            for r in all_recs:
                if r['score'] > 0.75:
                    if rng.random() > 0.1: # 90% chance to like it
                        relevant.add(r['id'])
                elif r['score'] > 0.5:
                    if rng.random() > 0.8: # 20% chance to like it
                        relevant.add(r['id'])
            
            self.test_candidates.append({
                "candidate_id": cand_id,
                "relevant_jobs": relevant
            })

    def evaluate(self, model, name: str) -> Dict[str, float]:
        hits = 0
        total_recs = 0
        unique_recommended = set()
        
        for query in self.test_candidates:
            cand_id = query["candidate_id"]
            relevant = query["relevant_jobs"]
            
            recs = model.recommend_jobs_for_candidate(cand_id, self.k)
            rec_ids = [r['id'] for r in recs]
            
            for rec_id in rec_ids:
                unique_recommended.add(rec_id)
                if rec_id in relevant:
                    hits += 1
            
            total_recs += len(recs)
            
        precision_at_k = hits / total_recs if total_recs > 0 else 0
        coverage = len(unique_recommended) / len(self.catalog.jobs_df)
        
        results = {
            "model": name,
            f"precision@{self.k}": precision_at_k,
            "coverage": coverage,
            "diversity": len(unique_recommended)
        }
        logger.info(f"Evaluated {name}: {results}")
        return results
