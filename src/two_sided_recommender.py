import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)

class CatalogManager:
    def __init__(self, random_state=42):
        self.rng = np.random.default_rng(random_state)
        self.locations = ['New York', 'San Francisco', 'Austin', 'Remote', 'London']
        self.seniorities = ['Entry Level', 'Mid Level', 'Senior', 'Director']
        self.skills_pool = ['Python', 'SQL', 'Java', 'React', 'AWS', 'GCP', 'Azure', 'C++', 'ML', 'Docker']
        
        self.candidates_df = None
        self.jobs_df = None
        
    def generate_catalogs(self, num_candidates=5000, num_jobs=2000):
        # Generate Candidates
        cand_data = []
        for i in range(num_candidates):
            cand_id = f"c_{i:05d}"
            loc = self.rng.choice(self.locations)
            sen = self.rng.choice(self.seniorities)
            num_skills = self.rng.integers(2, 6)
            skills = list(self.rng.choice(self.skills_pool, size=num_skills, replace=False))
            cand_data.append({
                "candidate_id": cand_id,
                "location": loc,
                "seniority": sen,
                "skills": skills
            })
        self.candidates_df = pd.DataFrame(cand_data)
        self.candidates_df.set_index("candidate_id", inplace=True)
        
        # Generate Jobs
        job_data = []
        for i in range(num_jobs):
            job_id = f"j_{i:05d}"
            loc = self.rng.choice(self.locations)
            sen = self.rng.choice(self.seniorities)
            num_skills = self.rng.integers(2, 6)
            skills = list(self.rng.choice(self.skills_pool, size=num_skills, replace=False))
            job_data.append({
                "job_id": job_id,
                "location": loc,
                "seniority": sen,
                "req_skills": skills,
                "popularity_score": self.rng.random()  # for baseline
            })
        self.jobs_df = pd.DataFrame(job_data)
        self.jobs_df.set_index("job_id", inplace=True)
        
        logger.info(f"Generated {num_candidates} candidates and {num_jobs} jobs.")

class TwoSidedRecommender:
    def __init__(self, catalog: CatalogManager):
        self.catalog = catalog
        self.seniority_map = {'Entry Level': 1, 'Mid Level': 2, 'Senior': 3, 'Director': 4}
        
        # Vectorized structures for fast scoring
        if self.catalog.jobs_df is not None:
            self._prepare_fast_scoring_jobs()
        if self.catalog.candidates_df is not None:
            self._prepare_fast_scoring_candidates()

    def _prepare_fast_scoring_jobs(self):
        # We prepare arrays to vectorize candidate -> all jobs scoring
        jobs = self.catalog.jobs_df
        self.job_ids = jobs.index.values
        self.job_locs = jobs['location'].values
        self.job_sens = jobs['seniority'].map(self.seniority_map).values
        # One-hot encoding of skills for fast intersection
        self.job_skills_matrix = np.zeros((len(jobs), len(self.catalog.skills_pool)))
        for i, skills in enumerate(jobs['req_skills']):
            for s in skills:
                idx = self.catalog.skills_pool.index(s)
                self.job_skills_matrix[i, idx] = 1.0
        # denominator for skill score
        self.job_skills_count = self.job_skills_matrix.sum(axis=1)
        # Avoid div by zero
        self.job_skills_count[self.job_skills_count == 0] = 1.0

    def _prepare_fast_scoring_candidates(self):
        cands = self.catalog.candidates_df
        self.cand_ids = cands.index.values
        self.cand_locs = cands['location'].values
        self.cand_sens = cands['seniority'].map(self.seniority_map).values
        self.cand_skills_matrix = np.zeros((len(cands), len(self.catalog.skills_pool)))
        for i, skills in enumerate(cands['skills']):
            for s in skills:
                idx = self.catalog.skills_pool.index(s)
                self.cand_skills_matrix[i, idx] = 1.0

    def _generate_explanation(self, loc_match, sen_diff, skill_score) -> str:
        reasons = []
        if loc_match:
            reasons.append("Location aligns.")
        if sen_diff == 0:
            reasons.append("Exact seniority match.")
        if skill_score > 0.5:
            reasons.append(f"Strong skill overlap ({int(skill_score*100)}%).")
            
        if not reasons:
            return "Partial match based on profile."
        return "Good match because: " + " ".join(reasons)

    def recommend_jobs_for_candidate(self, candidate_id: str, k: int = 10) -> List[Dict[str, Any]]:
        if candidate_id not in self.catalog.candidates_df.index:
            raise ValueError(f"Candidate {candidate_id} not found.")
            
        cand_row = self.catalog.candidates_df.loc[candidate_id]
        
        # Vectorized scoring
        c_loc = cand_row['location']
        c_sen = self.seniority_map[cand_row['seniority']]
        
        c_skills_vec = np.zeros(len(self.catalog.skills_pool))
        for s in cand_row['skills']:
            c_skills_vec[self.catalog.skills_pool.index(s)] = 1.0
            
        # Location match
        loc_match = (self.job_locs == c_loc) | (self.job_locs == 'Remote')
        
        # Seniority match
        sen_diff = c_sen - self.job_sens
        sen_score = np.where(sen_diff < 0, 0.2, np.where(sen_diff > 0, 0.5, 1.0))
        
        # Skills match
        intersection = self.job_skills_matrix.dot(c_skills_vec)
        skill_score = intersection / self.job_skills_count
        
        # Final scores
        scores = (0.5 * skill_score) + (0.3 * loc_match.astype(float)) + (0.2 * sen_score)
        
        # Top K
        top_k_idx = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_k_idx:
            job_id = self.job_ids[idx]
            explanation = self._generate_explanation(loc_match[idx], sen_diff[idx], skill_score[idx])
            results.append({
                "id": job_id,
                "score": float(scores[idx]),
                "explanation": explanation
            })
            
        return results
        
    def recommend_candidates_for_job(self, job_id: str, k: int = 10) -> List[Dict[str, Any]]:
        if job_id not in self.catalog.jobs_df.index:
            raise ValueError(f"Job {job_id} not found.")
            
        job_row = self.catalog.jobs_df.loc[job_id]
        
        # Vectorized scoring
        j_loc = job_row['location']
        j_sen = self.seniority_map[job_row['seniority']]
        
        j_skills_vec = np.zeros(len(self.catalog.skills_pool))
        for s in job_row['req_skills']:
            j_skills_vec[self.catalog.skills_pool.index(s)] = 1.0
        
        # Location match
        loc_match = (self.cand_locs == j_loc) | (j_loc == 'Remote')
        
        # Seniority match
        sen_diff = self.cand_sens - j_sen
        sen_score = np.where(sen_diff < 0, 0.2, np.where(sen_diff > 0, 0.5, 1.0))
        
        # Skills match
        j_skill_count = len(job_row['req_skills'])
        if j_skill_count == 0:
            skill_score = np.ones(len(self.cand_ids))
        else:
            intersection = self.cand_skills_matrix.dot(j_skills_vec)
            skill_score = intersection / j_skill_count
            
        # Final scores
        scores = (0.5 * skill_score) + (0.3 * loc_match.astype(float)) + (0.2 * sen_score)
        
        # Top K
        top_k_idx = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_k_idx:
            cand_id = self.cand_ids[idx]
            explanation = self._generate_explanation(loc_match[idx], sen_diff[idx], skill_score[idx])
            results.append({
                "id": cand_id,
                "score": float(scores[idx]),
                "explanation": explanation
            })
            
        return results

class BaselineRecommender:
    def __init__(self, catalog: CatalogManager):
        self.catalog = catalog
        
    def recommend_jobs_for_candidate(self, candidate_id: str, k: int = 10) -> List[Dict[str, Any]]:
        # Returns top jobs by popularity_score
        top_jobs = self.catalog.jobs_df.nlargest(k, "popularity_score")
        return [{"id": idx, "score": float(row["popularity_score"]), "explanation": "Popular job"} for idx, row in top_jobs.iterrows()]

    def recommend_candidates_for_job(self, job_id: str, k: int = 10) -> List[Dict[str, Any]]:
        # Just random candidates
        cand_ids = self.catalog.candidates_df.sample(k).index
        return [{"id": c_id, "score": 1.0, "explanation": "Random baseline"} for c_id in cand_ids]
