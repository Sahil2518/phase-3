import numpy as np
import logging
from typing import Dict, List
from src.semantic_search_engine import SemanticSearchEngine

logger = logging.getLogger(__name__)

class SearchEvaluator:
    def __init__(self, engine: SemanticSearchEngine):
        self.engine = engine
        # These queries use PARAPHRASES that do NOT appear in the corpus verbatim.
        # Keyword search will return zero or random results; semantic/LSA must bridge the gap.
        self.queries = [
            {
                "query": "orchestrating batch workloads and transforming raw data",
                # Resumes contain: etl, airflow, spark, data pipelines — NOT these exact words
                "target_topic": "data_engineering"
            },
            {
                "query": "building visual interfaces for end users",
                # Resumes contain: react, javascript, css, html — NOT these exact words
                "target_topic": "frontend_dev"
            },
            {
                "query": "designing distributed server-side services",
                # Resumes contain: java, spring, microservices, sql — NOT these exact words
                "target_topic": "backend_dev"
            },
            {
                "query": "training neural models for perception tasks",
                # Resumes contain: machine learning, deep learning, pytorch — NOT these exact words
                "target_topic": "ml_ai"
            }
        ]
        
    def evaluate(self, method: str, alpha: float = 0.8, k: int = 10) -> Dict[str, float]:
        """
        Evaluate semantic search vs keyword search.
        Since we want 'data pipelines' to match 'etl' and 'airflow' (which are in the data_engineering topic),
        our metric will be: % of top-K results that belong to the target_topic.
        This represents precision@k for the underlying conceptual category.
        """
        logger.info(f"Evaluating {method} search (alpha={alpha})...")
        
        total_precision = 0.0
        mrr = 0.0
        
        for q in self.queries:
            results = self.engine.search(q["query"], method=method, alpha=alpha, k=k)
            
            # If no results (keyword search might fail), precision is 0
            if not results:
                continue
                
            hits = 0
            first_hit_rank = 0
            
            for rank, res in enumerate(results, start=1):
                doc_id = res["doc_id"]
                doc_topic = self.engine.catalog.resumes_df.loc[doc_id, "topic"]
                if doc_topic == q["target_topic"]:
                    hits += 1
                    if first_hit_rank == 0:
                        first_hit_rank = rank
                        
            precision_at_k = hits / k
            total_precision += precision_at_k
            
            if first_hit_rank > 0:
                mrr += 1.0 / first_hit_rank
                
        avg_precision = total_precision / len(self.queries)
        avg_mrr = mrr / len(self.queries)
        
        logger.info(f"Results for {method}: Precision@{k}={avg_precision:.4f}, MRR={avg_mrr:.4f}")
        return {
            "method": method,
            f"precision@{k}": avg_precision,
            "mrr": avg_mrr
        }
