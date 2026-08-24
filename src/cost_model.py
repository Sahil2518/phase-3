"""
cost_model.py — PlaceMux Task 21: Cost Optimization & FinOps
=============================================================
Cost model for the intelligence layer (train + serve).

Design
------
We proxy real cloud compute cost using wall-clock execution time:
    - Serve cost:    $0.000010 per ms   (equiv. to ~$0.01/s of CPU)
    - Train cost:    $0.050000 per run  (flat rate for index rebuild)

These constants can be swapped for real billing data from cloud providers.

Unit economics exported via `to_dict()` for the Data-Analyst handoff.
"""

import time
import logging
import json
import os
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cost Constants (adjustable for real billing data)
# ---------------------------------------------------------------------------
COST_PER_MS_SERVE: float = 0.000010      # $0.00001 / ms of CPU
COST_PER_TRAIN_RUN: float = 0.050000     # $0.05 flat per index build
COST_PER_BATCH_ITEM: float = 0.0000025   # small discount for bulk shortlists


class CostRecord:
    """Single measured cost event."""

    def __init__(
        self,
        event_type: str,       # 'inference' | 'training' | 'shortlist'
        duration_ms: float,
        cost_usd: float,
        tag: str = "",
    ):
        self.event_type = event_type
        self.duration_ms = duration_ms
        self.cost_usd = cost_usd
        self.tag = tag          # e.g. 'baseline' or 'optimized'

    def __repr__(self) -> str:
        return (
            f"CostRecord(type={self.event_type}, "
            f"duration={self.duration_ms:.2f}ms, "
            f"cost=${self.cost_usd:.8f}, tag={self.tag})"
        )


class IntelligenceCostModel:
    """
    Tracks and computes the cost of the intelligence layer.

    Usage
    -----
    model = IntelligenceCostModel(tag="baseline")

    with model.measure_inference():
        results = engine.search(query, ...)

    print(model.summary())
    """

    def __init__(self, tag: str = ""):
        self.tag = tag
        self.records: List[CostRecord] = []
        self.train_runs: int = 0

    # ------------------------------------------------------------------
    # Context-manager helpers
    # ------------------------------------------------------------------

    class _Timer:
        """Simple wall-clock timer context manager."""
        def __enter__(self):
            self._start = time.perf_counter()
            return self

        def __exit__(self, *_):
            self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0

    def measure_inference(self):
        """Context manager that records a single inference event."""
        timer = self._Timer()

        class _Ctx:
            def __init__(inner_self):
                inner_self.timer = timer

            def __enter__(inner_self):
                timer.__enter__()
                return inner_self

            def __exit__(inner_self, *args):
                timer.__exit__(*args)
                dur = timer.elapsed_ms
                cost = dur * COST_PER_MS_SERVE
                rec = CostRecord("inference", dur, cost, self.tag)
                self.records.append(rec)
                logger.debug(rec)

        return _Ctx()

    def measure_shortlist(self, batch_size: int = 1):
        """Context manager that records a batch shortlist event."""
        timer = self._Timer()

        class _Ctx:
            def __init__(inner_self):
                inner_self.timer = timer

            def __enter__(inner_self):
                timer.__enter__()
                return inner_self

            def __exit__(inner_self, *args):
                timer.__exit__(*args)
                dur = timer.elapsed_ms
                # shortlist gets compute cost + per-item fee
                cost = dur * COST_PER_MS_SERVE + batch_size * COST_PER_BATCH_ITEM
                rec = CostRecord("shortlist", dur, cost, self.tag)
                self.records.append(rec)
                logger.debug(rec)

        return _Ctx()

    def record_training_run(self, duration_ms: float):
        """Record the cost of building / rebuilding the index."""
        cost = COST_PER_TRAIN_RUN + duration_ms * COST_PER_MS_SERVE
        rec = CostRecord("training", duration_ms, cost, self.tag)
        self.records.append(rec)
        self.train_runs += 1
        logger.info(f"Training run recorded: {rec}")

    # ------------------------------------------------------------------
    # Aggregations
    # ------------------------------------------------------------------

    def total_cost(self) -> float:
        return sum(r.cost_usd for r in self.records)

    def inference_records(self) -> List[CostRecord]:
        return [r for r in self.records if r.event_type == "inference"]

    def training_records(self) -> List[CostRecord]:
        return [r for r in self.records if r.event_type == "training"]

    def shortlist_records(self) -> List[CostRecord]:
        return [r for r in self.records if r.event_type == "shortlist"]

    def cost_per_1k_inferences(self) -> float:
        inf = self.inference_records()
        if not inf:
            return 0.0
        avg = sum(r.cost_usd for r in inf) / len(inf)
        return avg * 1000.0

    def avg_latency_ms(self) -> float:
        inf = self.inference_records()
        if not inf:
            return 0.0
        return sum(r.duration_ms for r in inf) / len(inf)

    def summary(self) -> Dict[str, Any]:
        inf = self.inference_records()
        train = self.training_records()
        sl = self.shortlist_records()
        return {
            "tag": self.tag,
            "total_inferences": len(inf),
            "total_training_runs": len(train),
            "total_shortlists": len(sl),
            "total_cost_usd": round(self.total_cost(), 8),
            "cost_per_1k_inferences_usd": round(self.cost_per_1k_inferences(), 6),
            "avg_inference_latency_ms": round(self.avg_latency_ms(), 4),
            "avg_train_cost_usd": (
                round(sum(r.cost_usd for r in train) / len(train), 6) if train else 0.0
            ),
            "constants": {
                "cost_per_ms_serve_usd": COST_PER_MS_SERVE,
                "cost_per_train_run_usd": COST_PER_TRAIN_RUN,
                "cost_per_batch_item_usd": COST_PER_BATCH_ITEM,
            },
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"IntelligenceCostModel(tag={s['tag']}, "
            f"inferences={s['total_inferences']}, "
            f"cost/1k=${s['cost_per_1k_inferences_usd']:.6f}, "
            f"avg_latency={s['avg_inference_latency_ms']:.2f}ms)"
        )


# ---------------------------------------------------------------------------
# Comparison Utility
# ---------------------------------------------------------------------------

def compare_cost_models(
    before: IntelligenceCostModel,
    after: IntelligenceCostModel,
) -> Dict[str, Any]:
    """
    Compare before/after cost models.  Quality must be assessed separately
    by the caller (e.g. by checking top-K overlap).
    """
    b = before.summary()
    a = after.summary()

    b_c1k = b["cost_per_1k_inferences_usd"]
    a_c1k = a["cost_per_1k_inferences_usd"]

    saving_pct = 0.0
    if b_c1k > 0:
        saving_pct = (1.0 - a_c1k / b_c1k) * 100.0

    b_lat = b["avg_inference_latency_ms"]
    a_lat = a["avg_inference_latency_ms"]

    lat_saving_pct = 0.0
    if b_lat > 0:
        lat_saving_pct = (1.0 - a_lat / b_lat) * 100.0

    return {
        "before": b,
        "after": a,
        "cost_per_1k_before_usd": b_c1k,
        "cost_per_1k_after_usd": a_c1k,
        "cost_saving_pct": round(saving_pct, 2),
        "latency_before_ms": b_lat,
        "latency_after_ms": a_lat,
        "latency_saving_pct": round(lat_saving_pct, 2),
        "verdict": (
            "PASS — material cost reduction achieved."
            if saving_pct >= 20
            else "FAIL — cost reduction below 20% threshold."
        ),
    }


def export_economics(
    comparison: Dict[str, Any],
    path: str = "economics_handoff.json",
) -> None:
    """Export unit economics to JSON for Data-Analyst handoff."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(comparison, f, indent=2)
    logger.info(f"Economics handoff written to {path}")
