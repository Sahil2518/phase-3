"""
enterprise_pilot_dataset.py -- PlaceMux Phase 3, Task 20
=========================================================
Generates the AcmeCorp enterprise pilot dataset:
  - 2,000 candidates x 600 jobs with demographic attributes
  - 5,000 position-biased interaction events
  - 80/20 train/test split stratified on query_id
Seed=42 throughout for full reproducibility.
"""

import os
import sys
import logging
import json
import numpy as np
import pandas as pd
from typing import Tuple

logger = logging.getLogger(__name__)

TENANT_ID = "acmecorp"
NUM_CANDIDATES = 2_000
NUM_JOBS = 600
NUM_INTERACTIONS = 5_000
RANDOM_STATE = 42

LOCATIONS = ["New York", "San Francisco", "Austin", "Remote", "London"]
LOCATION_TIERS = {
    "New York": "Tier1", "San Francisco": "Tier1", "London": "Tier1",
    "Austin": "Tier2", "Remote": "Tier2",
}
SENIORITIES = ["Entry Level", "Mid Level", "Senior", "Director"]
SENIORITY_MAP = {"Entry Level": 1, "Mid Level": 2, "Senior": 3, "Director": 4}
SKILLS_POOL = ["Python", "SQL", "Java", "React", "AWS", "GCP", "Azure", "C++", "ML", "Docker"]
GENDERS = ["M", "F", "NB"]


class EnterprisePilotDataset:
    """
    Generates and manages the AcmeCorp enterprise pilot dataset.

    Demographic attributes (gender, location_tier) enable fairness evaluation
    in Stage C. Position bias is baked into interactions so IPS correction
    in Stage B has something meaningful to address.

    Parameters
    ----------
    random_state : int
        RNG seed for full reproducibility.
    """

    def __init__(self, random_state: int = RANDOM_STATE):
        self.rng = np.random.default_rng(random_state)
        self.candidates = pd.DataFrame()
        self.jobs = pd.DataFrame()
        self.interactions = pd.DataFrame()
        self.train_interactions = pd.DataFrame()
        self.test_interactions = pd.DataFrame()

    def _generate_candidates(self) -> pd.DataFrame:
        """
        Create candidate catalogue with skills, location, seniority, gender.

        Returns
        -------
        pd.DataFrame
            Indexed by candidate_id.
        """
        records = []
        for i in range(NUM_CANDIDATES):
            loc = self.rng.choice(LOCATIONS)
            sen = self.rng.choice(SENIORITIES)
            gender = self.rng.choice(GENDERS, p=[0.48, 0.45, 0.07])
            n_skills = int(self.rng.integers(2, 7))
            skills = list(self.rng.choice(SKILLS_POOL, size=n_skills, replace=False))
            records.append({
                "candidate_id": f"c_{TENANT_ID}_{i:05d}",
                "location": loc,
                "location_tier": LOCATION_TIERS[loc],
                "seniority": sen,
                "seniority_level": SENIORITY_MAP[sen],
                "gender": gender,
                "skills": skills,
                "tenant_id": TENANT_ID,
            })
        df = pd.DataFrame(records).set_index("candidate_id")
        logger.info(f"Generated {len(df)} candidates.")
        return df

    def _generate_jobs(self) -> pd.DataFrame:
        """
        Create job catalogue with required skills, location, seniority.

        Returns
        -------
        pd.DataFrame
            Indexed by job_id.
        """
        records = []
        for i in range(NUM_JOBS):
            loc = self.rng.choice(LOCATIONS)
            sen = self.rng.choice(SENIORITIES)
            n_skills = int(self.rng.integers(2, 6))
            skills = list(self.rng.choice(SKILLS_POOL, size=n_skills, replace=False))
            records.append({
                "job_id": f"j_{TENANT_ID}_{i:05d}",
                "location": loc,
                "location_tier": LOCATION_TIERS[loc],
                "seniority": sen,
                "seniority_level": SENIORITY_MAP[sen],
                "req_skills": skills,
                "tenant_id": TENANT_ID,
                "popularity_score": float(self.rng.uniform(0.1, 1.0)),
            })
        df = pd.DataFrame(records).set_index("job_id")
        logger.info(f"Generated {len(df)} jobs.")
        return df

    @staticmethod
    def _jaccard(a: list, b: list) -> float:
        """
        Jaccard similarity between two skill lists.

        Parameters
        ----------
        a : list
        b : list

        Returns
        -------
        float
        """
        sa, sb = set(a), set(b)
        if not sa and not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    def _generate_interactions(self) -> pd.DataFrame:
        """
        Simulate 5,000 position-biased click events.

        Each query: pick a random candidate, score 50 random jobs, show
        top-10 ranked by oracle relevance, then sample clicks using a
        position-decay examination model P(exam|pos) = 1/sqrt(pos).

        Returns
        -------
        pd.DataFrame
            Columns: query_id, candidate_id, job_id, position, relevance, click.
        """
        cand_ids = self.candidates.index.tolist()
        job_ids = self.jobs.index.tolist()
        MAX_POS = 10
        exam_prob = {p: 1.0 / np.sqrt(p) for p in range(1, MAX_POS + 1)}

        records = []
        for q in range(NUM_INTERACTIONS):
            cid = self.rng.choice(cand_ids)
            cand = self.candidates.loc[cid]
            sample_jobs = self.rng.choice(job_ids, size=min(50, len(job_ids)), replace=False)

            scored = []
            for jid in sample_jobs:
                job = self.jobs.loc[jid]
                overlap = self._jaccard(cand["skills"], job["req_skills"])
                loc_ok = 1.0 if (cand["location"] == job["location"] or
                                  job["location"] == "Remote") else 0.0
                sen_diff = abs(cand["seniority_level"] - job["seniority_level"])
                sen_ok = max(0.0, 1.0 - 0.3 * sen_diff)
                rel = 0.5 * overlap + 0.3 * loc_ok + 0.2 * sen_ok
                scored.append((jid, rel))

            scored.sort(key=lambda x: x[1], reverse=True)
            for pos, (jid, rel) in enumerate(scored[:MAX_POS], start=1):
                p_click = exam_prob[pos] * (0.6 + 0.4 * rel)
                click = int(self.rng.random() < p_click)
                records.append({
                    "query_id": q,
                    "candidate_id": cid,
                    "job_id": jid,
                    "position": pos,
                    "relevance": round(rel, 4),
                    "click": click,
                })

        df = pd.DataFrame(records)
        logger.info(f"Generated {len(df)} interaction rows ({df['click'].sum()} clicks).")
        return df

    def _split_interactions(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        80/20 train/test split stratified by query_id.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
        """
        qids = self.interactions["query_id"].unique().copy()
        self.rng.shuffle(qids)
        split = int(len(qids) * 0.8)
        train_q, test_q = set(qids[:split]), set(qids[split:])
        train = self.interactions[self.interactions["query_id"].isin(train_q)].copy()
        test = self.interactions[self.interactions["query_id"].isin(test_q)].copy()
        logger.info(f"Split: {len(train_q)} train / {len(test_q)} test queries.")
        return train, test

    def generate(self) -> "EnterprisePilotDataset":
        """
        Run full dataset generation pipeline.

        Returns
        -------
        EnterprisePilotDataset
            self (for chaining).
        """
        logger.info("=== Generating AcmeCorp Enterprise Pilot Dataset ===")
        self.candidates = self._generate_candidates()
        self.jobs = self._generate_jobs()
        self.interactions = self._generate_interactions()
        self.train_interactions, self.test_interactions = self._split_interactions()
        logger.info("Dataset generation complete.")
        return self

    def save(self, out_dir: str = "logs") -> None:
        """
        Persist dataset CSVs for reproducibility.

        Parameters
        ----------
        out_dir : str
            Output directory path.
        """
        os.makedirs(out_dir, exist_ok=True)
        self.candidates.to_csv(os.path.join(out_dir, "task20_candidates.csv"))
        self.jobs.to_csv(os.path.join(out_dir, "task20_jobs.csv"))
        self.interactions.to_csv(os.path.join(out_dir, "task20_interactions.csv"), index=False)
        self.train_interactions.to_csv(os.path.join(out_dir, "task20_train.csv"), index=False)
        self.test_interactions.to_csv(os.path.join(out_dir, "task20_test.csv"), index=False)
        logger.info(f"Saved dataset CSVs to {out_dir}/")

    def summary(self) -> dict:
        """
        Return summary statistics dict.

        Returns
        -------
        dict
        """
        return {
            "tenant_id": TENANT_ID,
            "num_candidates": len(self.candidates),
            "num_jobs": len(self.jobs),
            "num_interactions": len(self.interactions),
            "num_clicks": int(self.interactions["click"].sum()),
            "ctr": round(float(self.interactions["click"].mean()), 4),
            "train_queries": int(self.train_interactions["query_id"].nunique()),
            "test_queries": int(self.test_interactions["query_id"].nunique()),
            "gender_distribution": self.candidates["gender"].value_counts().to_dict(),
            "seniority_distribution": self.candidates["seniority"].value_counts().to_dict(),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    ds = EnterprisePilotDataset().generate()
    ds.save()
    print(json.dumps(ds.summary(), indent=2))
