"""
cost_optimizer.py — PlaceMux Task 21: Cost Optimization & FinOps
=================================================================
Optimisations that reduce cost per inference / shortlist without
degrading quality:

1. ExactMatchCache
   ----------------
   - LRU cache keyed on (query, method, k).
   - Cache hit  → zero additional compute, same result.
   - Cache miss → runs full pipeline, stores result.
   - Quality:     100 % identical results for cache hits.

2. CascadeRouter
   ---------------
   - Runs the fast / cheap "keyword" search first.
   - If the best keyword score ≥ CONFIDENCE_THRESHOLD the result is
     served immediately (no semantic computation).
   - Only if keyword confidence is low does it escalate to the full
     semantic / hybrid search.
   - Degraded mode: if the expensive path fails, the keyword result is
     served with a warning flag rather than raising to the caller.

Quality Guarantee
-----------------
Both optimisations are evaluated against the baseline (unoptimised)
search in demo_task21.py.  The Top-K overlap metric is reported.
"""

import logging
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CACHE_MAX_SIZE: int = 512          # Maximum number of cached query results
CONFIDENCE_THRESHOLD: float = 0.55 # Keyword confidence to skip semantic step
DEFAULT_K: int = 10


# ---------------------------------------------------------------------------
# 1. Exact-Match Cache Wrapper
# ---------------------------------------------------------------------------

class ExactMatchCache:
    """
    Wraps any object that exposes a `.search(query, method, k)` method
    and caches results by (query, method, k).

    Usage
    -----
    cached_engine = ExactMatchCache(engine)
    results = cached_engine.search("machine learning", method="hybrid", k=10)
    """

    def __init__(self, engine, max_size: int = CACHE_MAX_SIZE):
        self._engine = engine
        self._cache: Dict[Tuple, List[Dict]] = {}
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def search(
        self,
        query: str,
        method: str = "hybrid",
        alpha: float = 0.8,
        k: int = DEFAULT_K,
    ) -> List[Dict]:
        key = (query.strip().lower(), method, k)
        if key in self._cache:
            self._hits += 1
            logger.debug(f"Cache HIT  for query='{query}' (hits={self._hits})")
            return self._cache[key]

        self._misses += 1
        logger.debug(f"Cache MISS for query='{query}' (misses={self._misses})")

        result = self._engine.search(query, method=method, alpha=alpha, k=k)

        # Simple FIFO eviction if full
        if len(self._cache) >= self._max_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[key] = result
        return result

    def cache_stats(self) -> Dict[str, Any]:
        return {
            "cache_size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_pct": round(self.hit_rate * 100, 2),
        }


# ---------------------------------------------------------------------------
# 2. Cascade Router
# ---------------------------------------------------------------------------

class CascadeRouter:
    """
    Two-level cascade:
      Level 1 (cheap)  — keyword-only search, O(sparse dot-product)
      Level 2 (costly) — full hybrid/semantic search, O(dense SVD)

    If the best keyword score is above CONFIDENCE_THRESHOLD, Level 2 is
    skipped entirely, saving the SVD transform cost.

    Graceful Degradation
    --------------------
    If the expensive (Level 2) path raises ANY exception, the router
    falls back to the keyword result and sets `degraded=True` in the
    response metadata.
    """

    def __init__(
        self,
        engine,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ):
        self._engine = engine
        self.threshold = confidence_threshold
        self._cascade_escalations = 0
        self._cascade_fast_exits = 0
        self._cascade_degraded = 0

    def search(
        self,
        query: str,
        method: str = "hybrid",   # target method when escalating
        alpha: float = 0.8,
        k: int = DEFAULT_K,
        _force_fail_expensive: bool = False,  # for failure-injection testing
    ) -> Dict[str, Any]:
        """
        Returns a dict with:
          results     — list of top-K result dicts
          route       — 'fast' | 'full' | 'degraded'
          top_score   — best keyword score seen
        """
        # ------------------------------------------------------------------
        # Level 1: keyword (cheap)
        # ------------------------------------------------------------------
        keyword_results = self._engine.search(query, method="keyword", k=k)

        top_score = keyword_results[0]["score"] if keyword_results else 0.0

        if top_score >= self.threshold:
            # High confidence — serve keyword result, skip expensive step
            self._cascade_fast_exits += 1
            logger.debug(
                f"Cascade FAST EXIT (score={top_score:.3f} >= {self.threshold})"
            )
            return {
                "results": keyword_results,
                "route": "fast",
                "top_score": top_score,
                "degraded": False,
            }

        # ------------------------------------------------------------------
        # Level 2: full / hybrid (expensive)
        # ------------------------------------------------------------------
        self._cascade_escalations += 1
        logger.debug(
            f"Cascade ESCALATE (score={top_score:.3f} < {self.threshold})"
        )

        try:
            if _force_fail_expensive:
                raise RuntimeError("Injected failure: expensive semantic path down.")

            full_results = self._engine.search(
                query, method=method, alpha=alpha, k=k
            )
            return {
                "results": full_results,
                "route": "full",
                "top_score": top_score,
                "degraded": False,
            }

        except Exception as exc:
            # Graceful degradation — fall back to keyword result with warning
            self._cascade_degraded += 1
            logger.warning(
                f"Cascade expensive path FAILED ({exc}). "
                f"Serving degraded keyword result."
            )
            return {
                "results": keyword_results,
                "route": "degraded",
                "top_score": top_score,
                "degraded": True,
                "error": str(exc),
            }

    def routing_stats(self) -> Dict[str, Any]:
        total = (
            self._cascade_fast_exits
            + self._cascade_escalations
            + self._cascade_degraded
        )
        return {
            "total_routed": total,
            "fast_exits": self._cascade_fast_exits,
            "full_escalations": self._cascade_escalations,
            "degraded_fallbacks": self._cascade_degraded,
            "fast_exit_rate_pct": (
                round(self._cascade_fast_exits / total * 100, 2) if total else 0.0
            ),
        }


# ---------------------------------------------------------------------------
# 3. Combined Optimized Engine
# ---------------------------------------------------------------------------

class OptimizedSearchEngine:
    """
    Chains ExactMatchCache → CascadeRouter → underlying engine.

    Call order per query:
      1. Check cache  → hit? return immediately (cheapest path).
      2. Route through cascade:
         a. keyword-only? return if confident (cheap path).
         b. full hybrid?  return (expensive path).
         c. degraded?     return keyword result + warning.
      3. Store result in cache.
    """

    def __init__(
        self,
        engine,
        cache_max_size: int = CACHE_MAX_SIZE,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ):
        self._engine = engine
        self._cache = ExactMatchCache(engine, max_size=cache_max_size)
        self._router = CascadeRouter(engine, confidence_threshold=confidence_threshold)

    def search(
        self,
        query: str,
        method: str = "hybrid",
        alpha: float = 0.8,
        k: int = DEFAULT_K,
        _force_fail_expensive: bool = False,
    ) -> List[Dict]:
        """
        Returns a plain list[dict] identical in shape to the raw engine.
        """
        cache_key = (query.strip().lower(), method, k)

        # Layer 1: cache
        if cache_key in self._cache._cache:
            self._cache._hits += 1
            logger.debug(f"OptimizedEngine cache HIT for '{query}'")
            return self._cache._cache[cache_key]
        self._cache._misses += 1

        # Layer 2: cascade
        routed = self._router.search(
            query,
            method=method,
            alpha=alpha,
            k=k,
            _force_fail_expensive=_force_fail_expensive,
        )
        results = routed["results"]

        # Store in cache
        if len(self._cache._cache) >= self._cache._max_size:
            oldest = next(iter(self._cache._cache))
            del self._cache._cache[oldest]
        self._cache._cache[cache_key] = results

        return results

    def stats(self) -> Dict[str, Any]:
        return {
            "cache": self._cache.cache_stats(),
            "routing": self._router.routing_stats(),
        }
