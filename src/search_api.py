import os
import time
import logging
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException

from src.semantic_search_engine import DocumentCatalog, SemanticSearchEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="PlaceMux Semantic Search API", version="1.0.0")

catalog = None
engine = None

@app.on_event("startup")
async def startup_event():
    global catalog, engine
    logger.info("Starting Semantic Search API...")
    catalog = DocumentCatalog(random_state=42)
    catalog.generate_catalog(num_docs=1000)
    
    engine = SemanticSearchEngine(catalog, n_components=4)
    engine.build_index()
    logger.info("Search Engine Ready.")

@app.get("/search")
async def search(query: str, method: str = "hybrid", alpha: float = 0.8, k: int = 10):
    start_time = time.time()
    try:
        MODEL_AVAILABLE = os.getenv("MODEL_AVAILABLE", "1") == "1"
        
        # Rule 7: API-level Model Unavailability Guard
        if not MODEL_AVAILABLE and method in ["semantic", "hybrid"]:
            logger.warning("Semantic model unavailable. Falling back to exact keyword search.")
            method = "keyword"
            
        results = engine.search(query=query, method=method, alpha=alpha, k=k)
        
        latency_ms = (time.time() - start_time) * 1000
        return {
            "query": query,
            "method_used": method,
            "latency_ms": latency_ms,
            "results": results
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")
