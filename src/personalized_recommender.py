import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any
from identity_manager import IdentityManager, UserContext

logger = logging.getLogger(__name__)

class OrgCatalogManager:
    """
    Manages job and candidate catalogs physically separated by Org ID.
    """
    def __init__(self, random_state=42):
        self.rng = np.random.default_rng(random_state)
        self.skills_pool = ['Python', 'SQL', 'Java', 'React', 'AWS']
        
        self.org_candidates: Dict[str, pd.DataFrame] = {}
        self.org_jobs: Dict[str, pd.DataFrame] = {}

    def generate_data(self, org_id: str, num_cands: int = 50, num_jobs: int = 10):
        """Generates mock synthetic data for a given org."""
        c_data = []
        for i in range(num_cands):
            c_data.append({
                "candidate_id": f"c_{org_id}_{i:03d}",
                "seniority": self.rng.integers(1, 5),
                "skills": list(self.rng.choice(self.skills_pool, size=2, replace=False))
            })
        df_c = pd.DataFrame(c_data).set_index("candidate_id")
        self.org_candidates[org_id] = df_c

        j_data = []
        for i in range(num_jobs):
            j_data.append({
                "job_id": f"j_{org_id}_{i:03d}",
                "req_seniority": self.rng.integers(1, 5),
                "req_skills": list(self.rng.choice(self.skills_pool, size=2, replace=False))
            })
        df_j = pd.DataFrame(j_data).set_index("job_id")
        self.org_jobs[org_id] = df_j

    def get_candidates(self, org_id: str) -> pd.DataFrame:
        if org_id not in self.org_candidates:
            raise ValueError(f"No catalog for Org {org_id}")
        return self.org_candidates[org_id]

    def get_jobs(self, org_id: str) -> pd.DataFrame:
        if org_id not in self.org_jobs:
            raise ValueError(f"No catalog for Org {org_id}")
        return self.org_jobs[org_id]


class PersonalizedRecommender:
    """
    Org-aware personalized recommender. Combines base rules with recruiter-specific signals.
    """
    def __init__(self, catalog: OrgCatalogManager, identity_manager: IdentityManager):
        self.catalog = catalog
        self.identity_manager = identity_manager

    def recommend_candidates_for_job(self, user_id: str, job_id: str, requested_org_id: str, k: int = 5) -> List[Dict[str, Any]]:
        """Scores candidates using the identity manager for strict isolation."""
        # Rule 7: Base API Guard
        if not user_id or not job_id:
            logger.warning("Empty user or job ID provided.")
            return []
            
        # 1. Identity Resolution & Isolation Check
        try:
            context = self.identity_manager.get_user_context(user_id)
        except PermissionError as e:
            logger.error(f"Auth failed: {e}")
            raise
            
        # The critical Isolation Trap: The API request asked for 'requested_org_id'
        # but we strictly enforce the context.org_id from the Identity layer.
        if context.org_id != requested_org_id:
            logger.critical(f"Data leak attempt! User {user_id} authorized for {context.org_id} attempted to access {requested_org_id}.")
            raise PermissionError(f"Access Denied. User not authorized for Org {requested_org_id}.")

        # 2. Data Retrieval
        org_id = context.org_id
        jobs = self.catalog.get_jobs(org_id)
        if job_id not in jobs.index:
            raise ValueError(f"Job {job_id} not found in Org {org_id}.")
            
        job = jobs.loc[job_id]
        cands = self.catalog.get_candidates(org_id)
        
        # 3. Personalized Scoring
        results = []
        signals = context.signals
        
        for cand_id, cand in cands.iterrows():
            # Base logic
            skill_overlap = len(set(cand['skills']).intersection(set(job['req_skills'])))
            base_skill_score = skill_overlap / 2.0
            
            sen_diff = abs(cand['seniority'] - job['req_seniority'])
            base_sen_score = 1.0 if sen_diff == 0 else 0.5 if sen_diff == 1 else 0.1
            
            # Apply Personalization Multipliers
            p_skill_score = base_skill_score * signals.skill_multiplier
            p_sen_score = base_sen_score * signals.seniority_multiplier
            
            # Rule 7: Bounds checking
            p_skill_score = max(0.0, min(p_skill_score, 1.0))
            p_sen_score = max(0.0, min(p_sen_score, 1.0))
            
            final_score = (0.6 * p_skill_score) + (0.4 * p_sen_score)
            results.append({"candidate_id": cand_id, "score": float(final_score)})
            
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]
