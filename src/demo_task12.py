import os
import sys
import time
import json
import logging
import asyncio

from src.two_sided_recommender import CatalogManager, TwoSidedRecommender, BaselineRecommender
from src.recommender_evaluator import RecommenderEvaluator
from src.recommender_api import app, startup_event
import asyncio

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def run_demo():
    print("=" * 80)
    print("PlaceMux Phase 3 - Task 12: Personalization & Recommendation Engine")
    print("=" * 80)

    # 1. Initialize and Generate Catalogs
    print("\n[Stage B] Building Two-Sided Recommendation Engine")
    catalog = CatalogManager(random_state=42)
    catalog.generate_catalogs(num_candidates=5000, num_jobs=2000)
    
    personalized = TwoSidedRecommender(catalog)
    baseline = BaselineRecommender(catalog)
    
    # 2. Evaluate Offline
    print("\n[Stage C] Offline Evaluation (Precision@K, Coverage, Diversity)")
    evaluator = RecommenderEvaluator(catalog, k=10)
    
    res_base = evaluator.evaluate(baseline, "Baseline (Popularity/Random)")
    res_pers = evaluator.evaluate(personalized, "Personalized Engine")
    
    print("\n--- Evaluation Gap ---")
    print(f"Precision@10: {res_base['precision@10']:.4f} vs {res_pers['precision@10']:.4f} (Lift: {(res_pers['precision@10']/max(0.0001, res_base['precision@10']))-1:.1%})")
    print(f"Coverage:     {res_base['coverage']:.1%} vs {res_pers['coverage']:.1%}")
    print(f"Diversity:    {res_base['diversity']} vs {res_pers['diversity']} unique items recommended")
    
    # 3. Test Serving Path and Explainability
    print("\n[Stage D] Serving Path & Explainability (Latency SLO)")
    
    # We call the endpoints directly to avoid TestClient dependencies
    asyncio.run(startup_event())
    
    cand_id = "c_00042"
    job_id = "j_00007"
    
    # A. Candidate -> Jobs
    print(f"\n--- Recommend Jobs for Candidate {cand_id} ---")
    cand_profile = catalog.candidates_df.loc[cand_id].to_dict()
    print(f"Profile: Location={cand_profile['location']}, Seniority={cand_profile['seniority']}, Skills={cand_profile['skills']}")
    
    from src.recommender_api import recommend_jobs, recommend_candidates
    
    start_time = time.time()
    data = asyncio.run(recommend_jobs(candidate_id=cand_id, k=3))
    latency = (time.time() - start_time) * 1000
    print(f"Latency: {data['latency_ms']:.2f}ms (End-to-End: {latency:.2f}ms)")
    print(f"Source: {data['source']}")
    for i, rec in enumerate(data['recommendations']):
        print(f"  {i+1}. Job {rec['id']} (Score: {rec['score']:.2f})")
        print(f"     Why? -> {rec['explanation']}")
    
    # B. Company -> Candidates
    print(f"\n--- Recommend Candidates for Job {job_id} ---")
    job_profile = catalog.jobs_df.loc[job_id].to_dict()
    print(f"Job Spec: Location={job_profile['location']}, Seniority={job_profile['seniority']}, Req Skills={job_profile['req_skills']}")
    
    start_time = time.time()
    data = asyncio.run(recommend_candidates(job_id=job_id, k=3))
    latency = (time.time() - start_time) * 1000
    print(f"Latency: {data['latency_ms']:.2f}ms (End-to-End: {latency:.2f}ms)")
    print(f"Source: {data['source']}")
    for i, rec in enumerate(data['recommendations']):
        print(f"  {i+1}. Candidate {rec['id']} (Score: {rec['score']:.2f})")
        print(f"     Why? -> {rec['explanation']}")
            
    # C. Failure Path
    print("\n--- Failure Path (Model Unavailable) ---")
    os.environ["MODEL_AVAILABLE"] = "0"
    start_time = time.time()
    data = asyncio.run(recommend_jobs(candidate_id=cand_id, k=3))
    latency = (time.time() - start_time) * 1000
    print(f"Latency: {data['latency_ms']:.2f}ms (End-to-End: {latency:.2f}ms)")
    print(f"Source: {data['source']} -> Gracefully degraded to baseline")
    for i, rec in enumerate(data['recommendations']):
        print(f"  {i+1}. Job {rec['id']} (Score: {rec['score']:.2f}) - {rec['explanation']}")
            
    print("\n" + "=" * 80)
    print("Demo Complete.")
    print("=" * 80)

if __name__ == "__main__":
    run_demo()
