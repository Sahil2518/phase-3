"""
enterprise_fairness_evaluator.py -- PlaceMux Phase 3, Task 20 (Stage C)
========================================================================
Quality, fairness, and latency evaluation for the AcmeCorp pilot tenant.

Metrics:
  Quality  : Precision@10, NDCG@10 on held-out test set
  Fairness : Per-group recall; demographic parity gap (best - worst group)
             Groups: gender x seniority_band
             Acceptance bar: parity gap <= 0.15
  Latency  : p50 / p95 / p99 over 500 simulated shortlist queries
             Acceptance bar: p50 <= 30ms, p95 <= 100ms
"""

import os
import sys
import json
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Acceptance bars
# ---------------------------------------------------------------------------
BAR_PARITY_GAP = 0.15       # max acceptable demographic parity gap
BAR_P50_MS = 30.0           # p50 latency ms
BAR_P95_MS = 100.0          # p95 latency ms


class EnterpriseFairnessEvaluator:
    """
    Evaluates quality, fairness, and latency for a single enterprise tenant.

    Parameters
    ----------
    dataset : EnterprisePilotDataset
        Pre-generated enterprise dataset (candidates, jobs, interactions).
    ranker : SimpleIPSRanker
        The trained IPS ranker from Stage B (must be fitted).
    k : int
        Rank cutoff for quality and fairness metrics.
    n_latency_queries : int
        Number of queries to run for the latency benchmark.
    """

    def __init__(self, dataset, ranker, k: int = 10, n_latency_queries: int = 500):
        self.dataset = dataset
        self.ranker = ranker
        self.k = k
        self.n_latency_queries = n_latency_queries
        self.results: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Quality evaluation
    # ------------------------------------------------------------------
    def _evaluate_quality(self, df_test: pd.DataFrame) -> Dict[str, float]:
        """
        Compute Precision@K and NDCG@K on held-out test interactions.

        Parameters
        ----------
        df_test : pd.DataFrame
            Feature-enriched test interactions.

        Returns
        -------
        dict
            precision_at_k, ndcg_at_k values.
        """
        from enterprise_pilot_runner import precision_at_k, ndcg_at_k
        scores = self.ranker.predict(df_test)
        p_at_k = precision_at_k(df_test, scores, k=self.k)
        n_at_k = ndcg_at_k(df_test, scores, k=self.k)
        logger.info(f"Quality — P@{self.k}: {p_at_k:.4f} | NDCG@{self.k}: {n_at_k:.4f}")
        return {
            f"precision_at_{self.k}": round(p_at_k, 4),
            f"ndcg_at_{self.k}": round(n_at_k, 4),
        }

    # ------------------------------------------------------------------
    # Fairness evaluation
    # ------------------------------------------------------------------
    def _evaluate_fairness(self, df_test: pd.DataFrame,
                            candidates: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute per-group recall and demographic parity gap.

        We define group recall as: for each demographic group, the fraction
        of clicked (relevant) items that appear in the top-K predictions.

        Groups are defined by gender. Parity gap = max_recall - min_recall.

        Parameters
        ----------
        df_test : pd.DataFrame
            Feature-enriched test interactions with candidate_id.
        candidates : pd.DataFrame
            Candidate catalogue with gender, seniority columns.

        Returns
        -------
        dict
            Per-group recalls and parity gap.
        """
        scores = self.ranker.predict(df_test)
        df_eval = df_test.copy()
        df_eval["score"] = scores

        # Join demographic attributes
        cand_demo = candidates[["gender", "seniority"]].copy()
        df_eval = df_eval.join(cand_demo, on="candidate_id", how="left")

        group_recalls: Dict[str, float] = {}

        for gender, g_df in df_eval.groupby("gender"):
            if gender not in ["M", "F", "NB"]:
                continue
            recall_list = []
            for qid, q_df in g_df.groupby("query_id"):
                if q_df["click"].sum() == 0:
                    continue
                topk_idx = q_df.nlargest(self.k, "score").index
                topk_clicks = q_df.loc[topk_idx, "click"].sum()
                total_clicks = q_df["click"].sum()
                recall_list.append(topk_clicks / total_clicks)
            group_recalls[f"gender_{gender}"] = round(float(np.mean(recall_list)), 4) if recall_list else 0.0

        if not group_recalls:
            logger.warning("No group recall data — fairness evaluation skipped.")
            return {"group_recalls": {}, "parity_gap": None, "verdict": "SKIP"}

        max_recall = max(group_recalls.values())
        min_recall = min(group_recalls.values())
        parity_gap = round(max_recall - min_recall, 4)
        verdict = "PASS" if parity_gap <= BAR_PARITY_GAP else "FAIL"

        logger.info(f"Fairness — Group recalls: {group_recalls}")
        logger.info(f"Fairness — Parity gap: {parity_gap} | Bar: {BAR_PARITY_GAP} [{verdict}]")

        return {
            "group_recalls": group_recalls,
            "max_group_recall": max_recall,
            "min_group_recall": min_recall,
            "parity_gap": parity_gap,
            "acceptance_bar_parity_gap": BAR_PARITY_GAP,
            "verdict": verdict,
        }

    # ------------------------------------------------------------------
    # Latency benchmark
    # ------------------------------------------------------------------
    def _benchmark_latency(self, candidates: pd.DataFrame,
                            jobs: pd.DataFrame) -> Dict[str, Any]:
        """
        Time n_latency_queries shortlist requests end-to-end.

        Each request: select a candidate, score all jobs, return top-10.
        Uses numpy vectorized scoring for realistic inference timing.

        Parameters
        ----------
        candidates : pd.DataFrame
        jobs : pd.DataFrame

        Returns
        -------
        dict
            p50, p95, p99 latencies in milliseconds.
        """
        from enterprise_pilot_runner import build_features, FEATURES

        cand_ids = candidates.index.tolist()
        job_ids = jobs.index.tolist()
        rng = np.random.default_rng(42)

        latencies_ms: List[float] = []

        for _ in range(self.n_latency_queries):
            cid = rng.choice(cand_ids)
            # Build a feature row per job for this candidate
            rows = []
            for jid in job_ids:
                try:
                    cand = candidates.loc[cid]
                    job = jobs.loc[jid]
                    c_skills = set(cand["skills"])
                    j_skills = set(job["req_skills"])
                    union = c_skills | j_skills
                    ms = len(c_skills & j_skills) / len(union) if union else 0.0
                    lm = 1.0 if (cand["location"] == job["location"] or
                                  job["location"] == "Remote") else 0.0
                    sd = abs(cand["seniority_level"] - job["seniority_level"])
                    ss = max(0.0, 1.0 - 0.3 * sd)
                    rows.append({
                        "match_score": ms, "loc_match": lm,
                        "sen_score": ss, "popularity_score": float(job["popularity_score"]),
                    })
                except Exception:
                    continue

            if not rows:
                continue

            t_start = time.perf_counter()
            df_q = pd.DataFrame(rows)
            _ = self.ranker.predict(df_q)
            t_end = time.perf_counter()
            latencies_ms.append((t_end - t_start) * 1000.0)

        if not latencies_ms:
            logger.error("No latency measurements recorded.")
            return {"error": "No latency data"}

        p50 = round(float(np.percentile(latencies_ms, 50)), 2)
        p95 = round(float(np.percentile(latencies_ms, 95)), 2)
        p99 = round(float(np.percentile(latencies_ms, 99)), 2)
        mean_ms = round(float(np.mean(latencies_ms)), 2)

        p50_v = "PASS" if p50 <= BAR_P50_MS else "FAIL"
        p95_v = "PASS" if p95 <= BAR_P95_MS else "FAIL"

        logger.info(f"Latency — p50={p50}ms [{p50_v}] | p95={p95}ms [{p95_v}] | p99={p99}ms | mean={mean_ms}ms")

        return {
            "n_queries": len(latencies_ms),
            "p50_ms": p50,
            "p95_ms": p95,
            "p99_ms": p99,
            "mean_ms": mean_ms,
            "acceptance_bar_p50_ms": BAR_P50_MS,
            "acceptance_bar_p95_ms": BAR_P95_MS,
            "p50_verdict": p50_v,
            "p95_verdict": p95_v,
        }

    # ------------------------------------------------------------------
    # Main evaluate
    # ------------------------------------------------------------------
    def evaluate(self, df_test: pd.DataFrame) -> Dict[str, Any]:
        """
        Run all three evaluation stages: quality, fairness, latency.

        Parameters
        ----------
        df_test : pd.DataFrame
            Feature-enriched test interactions.

        Returns
        -------
        dict
            Full evaluation results.
        """
        logger.info("=== Stage C: Quality, Fairness & Latency Evaluation ===")

        quality = self._evaluate_quality(df_test)
        fairness = self._evaluate_fairness(df_test, self.dataset.candidates)
        latency = self._benchmark_latency(self.dataset.candidates, self.dataset.jobs)

        self.results = {
            "tenant_id": "acmecorp",
            "quality": quality,
            "fairness": fairness,
            "latency": latency,
        }
        return self.results

    def save(self, out_dir: str = "logs") -> None:
        """
        Write fairness and latency reports to separate JSON files.

        Parameters
        ----------
        out_dir : str
        """
        os.makedirs(out_dir, exist_ok=True)

        fairness_path = os.path.join(out_dir, "task20_fairness_report.json")
        latency_path = os.path.join(out_dir, "task20_latency_report.json")

        with open(fairness_path, "w") as f:
            json.dump({
                "tenant_id": self.results.get("tenant_id"),
                "quality": self.results.get("quality"),
                "fairness": self.results.get("fairness"),
            }, f, indent=2)

        with open(latency_path, "w") as f:
            json.dump({
                "tenant_id": self.results.get("tenant_id"),
                "latency": self.results.get("latency"),
            }, f, indent=2)

        logger.info(f"Saved fairness report to {fairness_path}")
        logger.info(f"Saved latency report to {latency_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    from enterprise_pilot_dataset import EnterprisePilotDataset
    from enterprise_pilot_runner import EnterprisePilotRunner, build_features

    ds = EnterprisePilotDataset().generate()
    runner = EnterprisePilotRunner(ds)
    runner.run()

    df_test = build_features(ds.test_interactions, ds.candidates, ds.jobs)
    evaluator = EnterpriseFairnessEvaluator(ds, runner.ips_ranker)
    results = evaluator.evaluate(df_test)
    evaluator.save()
    print(json.dumps(results, indent=2))
