import os
import time
import logging
import asyncio

from src.semantic_search_engine import DocumentCatalog, SemanticSearchEngine
from src.search_evaluator import SearchEvaluator
from src.search_api import app, startup_event, search

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def run_demo():
    print("=" * 80)
    print("PlaceMux Phase 3 - Task 13: Semantic Search & Vector Retrieval")
    print("=" * 80)

    # 1. Initialize and Generate Catalogs
    print("\n[Stage B] Building Embedding & Vector Index")
    catalog = DocumentCatalog(random_state=42)
    catalog.generate_catalog(num_docs=1000)
    
    engine = SemanticSearchEngine(catalog, n_components=4)
    engine.build_index()
    
    # 2. Evaluate Offline
    print("\n[Stage C & D] Offline Evaluation (Keyword vs Semantic vs Hybrid)")
    evaluator = SearchEvaluator(engine)
    
    res_kw = evaluator.evaluate(method="keyword", k=5)
    res_sem = evaluator.evaluate(method="semantic", k=5)
    res_hyb = evaluator.evaluate(method="hybrid", alpha=0.8, k=5)
    
    print("\n--- Evaluation Results (MRR) ---")
    print(f"Keyword Search MRR:  {res_kw['mrr']:.4f}")
    print(f"Semantic Search MRR: {res_sem['mrr']:.4f}")
    print(f"Hybrid Search MRR:   {res_hyb['mrr']:.4f}")
    
    print("\n--- Evaluation Results (Precision@5) ---")
    print(f"Keyword Precision@5:  {res_kw['precision@5']:.4f}")
    print(f"Semantic Precision@5: {res_sem['precision@5']:.4f}")
    print(f"Hybrid Precision@5:   {res_hyb['precision@5']:.4f}")
    
    # 3. Test Serving Path and Explainability
    print("\n[Stage E] Live Queries & Serving Path")
    
    # We call the endpoints directly
    asyncio.run(startup_event())
    
    query = "orchestrating batch workloads and transforming raw data"
    
    print(f"\n--- Query: '{query}' [Method: keyword] ---")
    data_kw = asyncio.run(search(query=query, method="keyword", k=3))
    print(f"Latency: {data_kw['latency_ms']:.2f}ms")
    for i, rec in enumerate(data_kw['results']):
        print(f"  {i+1}. Doc {rec['doc_id']} -> \"{rec['text']}\"")
        print(f"     Explain: {rec['explanation']}")

    print(f"\n--- Query: '{query}' [Method: semantic] ---")
    data_sem = asyncio.run(search(query=query, method="semantic", k=3))
    print(f"Latency: {data_sem['latency_ms']:.2f}ms")
    for i, rec in enumerate(data_sem['results']):
        print(f"  {i+1}. Doc {rec['doc_id']} -> \"{rec['text']}\"")
        print(f"     Explain: {rec['explanation']}")
        
    print(f"\n--- Query: '{query}' [Method: hybrid] ---")
    data_hyb = asyncio.run(search(query=query, method="hybrid", k=3))
    print(f"Latency: {data_hyb['latency_ms']:.2f}ms")
    for i, rec in enumerate(data_hyb['results']):
        print(f"  {i+1}. Doc {rec['doc_id']} -> \"{rec['text']}\"")
        print(f"     Explain: {rec['explanation']}")
            
    # 4. Failure Path
    print("\n--- Failure Path (Semantic Model Unavailable) ---")
    os.environ["MODEL_AVAILABLE"] = "0"
    data_fail = asyncio.run(search(query=query, method="hybrid", k=3))
    print(f"Requested: hybrid | Actual Used: {data_fail['method_used']} -> Gracefully degraded!")
    for i, rec in enumerate(data_fail['results']):
        print(f"  {i+1}. Doc {rec['doc_id']} -> \"{rec['text']}\"")
        print(f"     Explain: {rec['explanation']}")
            
    print("\n" + "=" * 80)
    print("Demo Complete.")
    print("=" * 80)

if __name__ == "__main__":
    run_demo()
