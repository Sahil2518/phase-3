import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any
from tenant_manager import TenantManager, TenantConfig

logger = logging.getLogger(__name__)

class TenantCatalogManager:
    """
    Manages data catalogs strictly separated by tenant_id to prevent cross-contamination.
    """
    def __init__(self, random_state=42):
        self.rng = np.random.default_rng(random_state)
        self.locations = ['New York', 'San Francisco', 'Austin', 'Remote', 'London']
        self.seniorities = ['Entry Level', 'Mid Level', 'Senior', 'Director']
        self.skills_pool = ['Python', 'SQL', 'Java', 'React', 'AWS', 'GCP', 'Azure', 'C++', 'ML', 'Docker']
        
        # Catalogs keyed by tenant_id -> DataFrame
        self.tenant_candidates: Dict[str, pd.DataFrame] = {}
        self.tenant_jobs: Dict[str, pd.DataFrame] = {}

    def generate_tenant_data(self, tenant_id: str, num_candidates: int = 1000, num_jobs: int = 500):
        """
        Generates simulated candidates and jobs explicitly bound to a tenant.
        
        Parameters
        ----------
        tenant_id : str
            The identifier for the tenant.
        num_candidates : int
            Number of candidates to generate.
        num_jobs : int
            Number of jobs to generate.
        """
        cand_data = []
        for i in range(num_candidates):
            cand_id = f"c_{tenant_id}_{i:05d}"
            loc = self.rng.choice(self.locations)
            sen = self.rng.choice(self.seniorities)
            skills = list(self.rng.choice(self.skills_pool, size=self.rng.integers(2, 6), replace=False))
            cand_data.append({
                "candidate_id": cand_id,
                "location": loc,
                "seniority": sen,
                "skills": skills,
                "tenant_id": tenant_id
            })
        df_c = pd.DataFrame(cand_data)
        df_c.set_index("candidate_id", inplace=True)
        self.tenant_candidates[tenant_id] = df_c

        job_data = []
        for i in range(num_jobs):
            job_id = f"j_{tenant_id}_{i:05d}"
            loc = self.rng.choice(self.locations)
            sen = self.rng.choice(self.seniorities)
            skills = list(self.rng.choice(self.skills_pool, size=self.rng.integers(2, 6), replace=False))
            job_data.append({
                "job_id": job_id,
                "location": loc,
                "seniority": sen,
                "req_skills": skills,
                "tenant_id": tenant_id,
                "popularity_score": self.rng.random()
            })
        df_j = pd.DataFrame(job_data)
        df_j.set_index("job_id", inplace=True)
        self.tenant_jobs[tenant_id] = df_j

        logger.info(f"Generated data for tenant {tenant_id}: {num_candidates} cands, {num_jobs} jobs.")

    def get_candidates(self, tenant_id: str) -> pd.DataFrame:
        """Retrieve isolated candidates for a tenant."""
        if tenant_id not in self.tenant_candidates:
            raise ValueError(f"No candidate data for tenant {tenant_id}")
        return self.tenant_candidates[tenant_id]

    def get_jobs(self, tenant_id: str) -> pd.DataFrame:
        """Retrieve isolated jobs for a tenant."""
        if tenant_id not in self.tenant_jobs:
            raise ValueError(f"No job data for tenant {tenant_id}")
        return self.tenant_jobs[tenant_id]

class MultiTenantRecommender:
    """
    Tenant-scoped inference engine with strict data isolation and dynamic weighting.
    """
    def __init__(self, catalog: TenantCatalogManager, tenant_manager: TenantManager):
        self.catalog = catalog
        self.tenant_manager = tenant_manager
        self.seniority_map = {'Entry Level': 1, 'Mid Level': 2, 'Senior': 3, 'Director': 4}
        # Pre-compiled vectorized structures mapped by tenant
        self.tenant_job_vectors = {}
        self.tenant_cand_vectors = {}

    def compile_tenant(self, tenant_id: str):
        """
        Pre-computes vectorized arrays explicitly isolated by tenant_id.
        """
        jobs = self.catalog.get_jobs(tenant_id)
        cands = self.catalog.get_candidates(tenant_id)

        # Build Job Vectors
        job_skills_mat = np.zeros((len(jobs), len(self.catalog.skills_pool)))
        for i, skills in enumerate(jobs['req_skills']):
            for s in skills:
                job_skills_mat[i, self.catalog.skills_pool.index(s)] = 1.0
        job_skills_count = job_skills_mat.sum(axis=1)
        job_skills_count[job_skills_count == 0] = 1.0

        self.tenant_job_vectors[tenant_id] = {
            'ids': jobs.index.values,
            'locs': jobs['location'].values,
            'sens': jobs['seniority'].map(self.seniority_map).values,
            'skills_mat': job_skills_mat,
            'skills_count': job_skills_count
        }

        # Build Candidate Vectors
        cand_skills_mat = np.zeros((len(cands), len(self.catalog.skills_pool)))
        for i, skills in enumerate(cands['skills']):
            for s in skills:
                cand_skills_mat[i, self.catalog.skills_pool.index(s)] = 1.0

        self.tenant_cand_vectors[tenant_id] = {
            'ids': cands.index.values,
            'locs': cands['location'].values,
            'sens': cands['seniority'].map(self.seniority_map).values,
            'skills_mat': cand_skills_mat
        }
        logger.info(f"Compiled vectorized matrices for tenant: {tenant_id}")

    def recommend_jobs_for_candidate(self, tenant_id: str, candidate_id: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Predict jobs for a candidate dynamically using the tenant's configuration.
        Ensures strict data isolation.
        
        Parameters
        ----------
        tenant_id : str
            The identifier of the tenant performing the query.
        candidate_id : str
            The identifier of the candidate.
        k : int
            Number of top recommendations to return.
            
        Returns
        -------
        List[Dict[str, Any]]
            A list of matching job recommendations.
        """
        # Rule 7: Guard against None models (or missing dependencies)
        if self.catalog is None or self.tenant_manager is None:
            raise ValueError("Cannot predict: recommender dependencies are uninitialized.")

        # Guard against empty inputs
        if not tenant_id or not candidate_id:
            logger.warning("Empty input provided. Returning empty response.")
            return []

        # Strict Data Isolation Check
        cands = self.catalog.get_candidates(tenant_id)
        if candidate_id not in cands.index:
            raise ValueError(f"Candidate {candidate_id} not found in tenant {tenant_id}'s data scope. Access Denied.")
            
        cand_row = cands.loc[candidate_id]
        
        # Dynamic Configuration Loading (No code branches)
        config = self.tenant_manager.get_config(tenant_id)
        
        # Ensure compiled arrays exist
        if tenant_id not in self.tenant_job_vectors:
            self.compile_tenant(tenant_id)
            
        vecs = self.tenant_job_vectors[tenant_id]
        
        # Vectorized scoring
        c_loc = cand_row['location']
        c_sen = self.seniority_map[cand_row['seniority']]
        
        c_skills_vec = np.zeros(len(self.catalog.skills_pool))
        for s in cand_row['skills']:
            c_skills_vec[self.catalog.skills_pool.index(s)] = 1.0
            
        loc_match = (vecs['locs'] == c_loc) | (vecs['locs'] == 'Remote')
        sen_diff = c_sen - vecs['sens']
        sen_score = np.where(sen_diff < 0, 0.2, np.where(sen_diff > 0, 0.5, 1.0))
        
        # Rule 7: Guard against division by zero (handled by skills_count having min 1.0)
        skill_score = vecs['skills_mat'].dot(c_skills_vec) / vecs['skills_count']
        
        # Apply tenant-specific weights
        raw_scores = (
            (config.skill_weight * skill_score) +
            (config.location_weight * loc_match.astype(float)) +
            (config.seniority_weight * sen_score)
        )
        
        # Rule 7: Check for NaNs
        if np.isnan(raw_scores).any():
            logger.warning("Invalid raw score detected (NaN). Replacing with 0.0")
            raw_scores = np.nan_to_num(raw_scores, nan=0.0)

        # Apply Threshold
        valid_idx = np.where(raw_scores >= config.match_threshold)[0]
        
        results = []
        for idx in valid_idx:
            job_id = vecs['ids'][idx]
            results.append({
                "id": job_id,
                "score": float(np.clip(raw_scores[idx], 0.0, 1.0)),
                "tenant_id": tenant_id
            })
            
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]
