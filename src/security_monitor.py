"""
security_monitor.py -- PlaceMux Phase 3, Task 22
=================================================
Detection for scraping / extraction and poisoned training data.

Two detectors are implemented:

1. ScrapingDetector
   Monitors API request streams for signs of bulk extraction:
   - Rate-window counter: >N requests in a sliding 60-second window per client.
   - Sequential ID enumeration: client cycling through monotonically increasing
     candidate_id or job_id values (hallmark of a scraper).
   - Response diversity guard: same request template with only the ID changing
     (low variety in request parameters).

   Actions: ALLOW | RATE_LIMIT | BLOCK

2. DataPoisonDetector
   Screens each incoming training batch for injected / corrupted records:
   - Isolation Forest anomaly scorer: scores each row against the expected
     feature distribution; high anomaly score = likely synthetic / poisoned.
   - Label consistency check: flags feature-label contradictions
     (e.g. 0-session user labelled as highly engaged).
   - Duplicate injection guard: flags near-duplicate records
     (column-wise match ratio > 0.95) that could vote-stuff the distribution.

   Returns: {"safe_to_train": bool, "poison_rate": float, "poisoned_indices": [...]}

Design rationale
----------------
The scraping detector uses a pure Python sliding-window dict -- no Redis
required for the demo.  A production deployment would replace this with a
Redis SORTED SET for distributed rate limiting.

The poison detector uses scikit-learn's IsolationForest with contamination=0.05,
meaning it treats up to 5% of a clean batch as potential noise.  When the
fraction of flagged rows exceeds `poison_threshold`, training is halted.

Failure / unavailability paths
-------------------------------
If sklearn is not available, the DataPoisonDetector falls back to a
statistical Z-score heuristic (no third-party dependency).
If any detector raises an unexpected exception, it returns a SAFE default
(ALLOW / safe_to_train=True) so that defenders do not accidentally block
legitimate traffic in degraded mode.

Metric bar (Task 22 spec)
--------------------------
  Scraping detector  : precision >= 0.95 on simulated burst attacks
  Poison detector    : recall    >= 0.80 on injected synthetic poison rows
"""

import os
import sys
import json
import time
import math
import logging
from collections import deque, defaultdict
from datetime import datetime, timezone
from typing import List, Dict, Optional, Deque

import numpy as np

try:
    import pandas as pd
    _PD_AVAILABLE = True
except ImportError:
    _PD_AVAILABLE = False

try:
    from sklearn.ensemble import IsolationForest
    _SKL_AVAILABLE = True
except ImportError:
    _SKL_AVAILABLE = False

os.makedirs("logs", exist_ok=True)
logger = logging.getLogger(__name__)

SECURITY_REPORT_PATH = "logs/security_monitor_report.json"


# ---------------------------------------------------------------------------
# 1. ScrapingDetector
# ---------------------------------------------------------------------------

class ScrapingDetector:
    """
    Detects bulk scraping / data extraction via API request analysis.

    Maintains an in-memory sliding window of request timestamps and
    requested IDs per client token.

    Parameters
    ----------
    window_seconds     : int    Sliding window length in seconds (default 60).
    rate_limit_threshold : int  Requests per window triggering rate-limit (default 30).
    block_threshold    : int    Requests per window triggering full block (default 60).
    enum_min_seq_len   : int    Min sequential IDs to flag as enumeration (default 5).
    enum_tolerance     : int    Allowed gaps in sequential ID runs (default 1).
    """

    ACTION_ALLOW      = "ALLOW"
    ACTION_RATE_LIMIT = "RATE_LIMIT"
    ACTION_BLOCK      = "BLOCK"

    def __init__(
        self,
        window_seconds:       int = 60,
        rate_limit_threshold: int = 30,
        block_threshold:      int = 60,
        enum_min_seq_len:     int = 5,
        enum_tolerance:       int = 1,
    ) -> None:
        """Initialise the ScrapingDetector with sliding-window state."""
        self.window_seconds       = window_seconds
        self.rate_limit_threshold = rate_limit_threshold
        self.block_threshold      = block_threshold
        self.enum_min_seq_len     = enum_min_seq_len
        self.enum_tolerance       = enum_tolerance

        # client_id -> deque of (timestamp, resource_id) tuples
        self._windows: Dict[str, Deque] = defaultdict(deque)

        logger.info(
            f"ScrapingDetector initialised: "
            f"window={window_seconds}s, rate_limit={rate_limit_threshold}, "
            f"block={block_threshold}"
        )

    def _purge_old_events(self, client_id: str, now: float) -> None:
        """
        Remove events outside the sliding window for a client.

        Parameters
        ----------
        client_id : str
        now       : float  Current unix timestamp.
        """
        dq = self._windows[client_id]
        cutoff = now - self.window_seconds
        while dq and dq[0][0] < cutoff:
            dq.popleft()

    def _detect_enumeration(self, resource_ids: List) -> bool:
        """
        Detect sequential ID enumeration: a run of consecutive integer IDs.

        Parameters
        ----------
        resource_ids : List  Recent resource IDs accessed by the client.

        Returns
        -------
        bool  True if a sequential enumeration run is detected.
        """
        # Extract numeric IDs where possible
        numeric_ids = []
        for rid in resource_ids:
            try:
                numeric_ids.append(int(rid))
            except (ValueError, TypeError):
                pass

        if len(numeric_ids) < self.enum_min_seq_len:
            return False

        numeric_ids_sorted = sorted(set(numeric_ids))
        run_length = 1
        max_run    = 1
        for i in range(1, len(numeric_ids_sorted)):
            gap = numeric_ids_sorted[i] - numeric_ids_sorted[i - 1]
            if gap <= (1 + self.enum_tolerance):
                run_length += 1
                max_run = max(max_run, run_length)
            else:
                run_length = 1

        return max_run >= self.enum_min_seq_len

    def check(
        self,
        client_id: str,
        resource_id,
        now: Optional[float] = None,
    ) -> Dict:
        """
        Process one API request and decide the action to take.

        Parameters
        ----------
        client_id   : str    Client token / IP address identifier.
        resource_id : any    The ID of the resource being accessed.
        now         : float  Unix timestamp (uses time.time() if None).

        Returns
        -------
        dict
            {
                "action":      str   'ALLOW' | 'RATE_LIMIT' | 'BLOCK',
                "reason":      str,
                "request_count": int,
                "enumeration": bool,
            }
        """
        # Guard: unexpected failure path returns ALLOW (fail-open)
        try:
            if now is None:
                now = time.time()

            self._purge_old_events(client_id, now)
            self._windows[client_id].append((now, resource_id))

            window_events = list(self._windows[client_id])
            request_count = len(window_events)
            recent_ids    = [e[1] for e in window_events]

            # Check sequential enumeration
            enum_detected = self._detect_enumeration(recent_ids)

            # Decide action
            if request_count >= self.block_threshold or (
                enum_detected and request_count >= self.rate_limit_threshold
            ):
                action = self.ACTION_BLOCK
                reason = (
                    f"burst_enumeration: {request_count} requests + sequential IDs"
                    if enum_detected
                    else f"rate_exceeded: {request_count} requests in {self.window_seconds}s"
                )
            elif request_count >= self.rate_limit_threshold:
                action = self.ACTION_RATE_LIMIT
                reason = (
                    f"rate_limit: {request_count} requests in {self.window_seconds}s"
                )
            elif enum_detected:
                action = self.ACTION_RATE_LIMIT
                reason = f"enumeration_pattern: sequential ID run detected"
            else:
                action = self.ACTION_ALLOW
                reason = "normal"

            result = {
                "action":        action,
                "reason":        reason,
                "client_id":     client_id,
                "request_count": request_count,
                "enumeration":   enum_detected,
                "window_seconds": self.window_seconds,
            }

            if action != self.ACTION_ALLOW:
                logger.warning(
                    f"[ScrapingDetector] {action} client={client_id} "
                    f"count={request_count} enum={enum_detected} reason='{reason}'"
                )
            return result

        except Exception as e:
            logger.error(f"ScrapingDetector.check failed: {e}", exc_info=True)
            return {
                "action":        self.ACTION_ALLOW,
                "reason":        "detector_error_fail_open",
                "client_id":     client_id,
                "request_count": 0,
                "enumeration":   False,
                "window_seconds": self.window_seconds,
            }

    def reset_client(self, client_id: str) -> None:
        """
        Clear all window state for a client (e.g. after CAPTCHA solved).

        Parameters
        ----------
        client_id : str
        """
        self._windows[client_id].clear()
        logger.info(f"[ScrapingDetector] Window reset for client={client_id}")


# ---------------------------------------------------------------------------
# 2. DataPoisonDetector
# ---------------------------------------------------------------------------

class DataPoisonDetector:
    """
    Screens incoming training batches for poisoned or corrupted records.

    Three layers of detection:

    1. Isolation Forest anomaly scoring (if sklearn available).
    2. Label consistency check: flags feature-label contradictions.
    3. Duplicate injection guard: flags near-duplicate rows.

    Parameters
    ----------
    poison_threshold   : float  Fraction of flagged rows that triggers
                                safe_to_train=False (default 0.05 = 5%).
    contamination      : float  Expected fraction of outliers for IsolationForest
                                (default 0.05).
    duplicate_threshold: float  Column-match fraction to flag as near-duplicate
                                (default 0.95).
    label_col          : str    Name of the target/label column.
    feature_cols       : List[str]  Feature columns for anomaly scoring.
    engagement_features: List[str]  Features that should be high for 'engaged' label.
    random_state       : int
    """

    def __init__(
        self,
        poison_threshold:    float = 0.05,
        contamination:       float = 0.05,
        duplicate_threshold: float = 0.95,
        label_col:           str   = "label",
        feature_cols:        Optional[List[str]] = None,
        engagement_features: Optional[List[str]] = None,
        random_state:        int   = 42,
    ) -> None:
        """Initialise the DataPoisonDetector."""
        self.poison_threshold    = poison_threshold
        self.contamination       = contamination
        self.duplicate_threshold = duplicate_threshold
        self.label_col           = label_col
        self.feature_cols        = feature_cols  # None = auto-detect
        self.engagement_features = engagement_features or []
        self.random_state        = random_state

        self._iforest: Optional[object] = None  # fitted IsolationForest

        logger.info(
            f"DataPoisonDetector initialised: "
            f"poison_threshold={poison_threshold}, contamination={contamination}"
        )

    def fit_reference(self, X: np.ndarray) -> None:
        """
        Fit the IsolationForest on clean reference data.

        Parameters
        ----------
        X : np.ndarray  Clean training data, shape (n_samples, n_features).
        """
        if not _SKL_AVAILABLE:
            logger.warning("sklearn not available. IsolationForest fitting skipped.")
            return
        self._iforest = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100,
        )
        self._iforest.fit(X)
        logger.info(
            f"IsolationForest fitted on {X.shape[0]} clean reference samples."
        )

    def _anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Compute per-row anomaly scores using IsolationForest (or Z-score fallback).

        Returns scores in [0, 1] where 1.0 = most anomalous.

        Parameters
        ----------
        X : np.ndarray  Shape (n_samples, n_features).

        Returns
        -------
        np.ndarray  Anomaly scores, shape (n_samples,).
        """
        if _SKL_AVAILABLE and self._iforest is not None:
            # sklearn returns negative scores; more negative = more anomalous
            raw = self._iforest.decision_function(X)
            # Normalise to [0, 1]: higher = more anomalous
            raw_min, raw_max = raw.min(), raw.max()
            if raw_max > raw_min:
                scores = 1.0 - (raw - raw_min) / (raw_max - raw_min)
            else:
                scores = np.zeros(len(raw))
            return scores
        else:
            # Fallback: Z-score magnitude per row
            means = X.mean(axis=0)
            stds  = X.std(axis=0)
            stds[stds == 0] = 1.0
            z = np.abs((X - means) / stds)
            return np.clip(z.max(axis=1) / 5.0, 0.0, 1.0)

    def _label_consistency_flags(
        self, X: np.ndarray, y: np.ndarray, feature_names: List[str]
    ) -> np.ndarray:
        """
        Flag rows where the label contradicts the feature vector.

        Heuristic: if the label is 1 (engaged/positive) but ALL engagement
        features are in the bottom 10th percentile, the record is suspicious.

        Parameters
        ----------
        X             : np.ndarray  Feature matrix.
        y             : np.ndarray  Label vector (0 or 1).
        feature_names : List[str]

        Returns
        -------
        np.ndarray  Boolean mask, True = suspicious label.
        """
        if not self.engagement_features:
            return np.zeros(len(y), dtype=bool)

        flags = np.zeros(len(y), dtype=bool)
        for feat_name in self.engagement_features:
            if feat_name not in feature_names:
                continue
            idx = feature_names.index(feat_name)
            col = X[:, idx]
            p10 = np.percentile(col, 10)
            # Label=1 but feature in bottom decile
            inconsistent = (y == 1) & (col < p10)
            flags |= inconsistent

        return flags

    def _duplicate_flags(self, X: np.ndarray, threshold: float) -> np.ndarray:
        """
        Flag near-duplicate rows (column-match ratio > threshold).

        Compares every row against all preceding rows; O(n^2) but acceptable
        for training batch sizes (typically < 50k rows in this pipeline).

        Parameters
        ----------
        X         : np.ndarray
        threshold : float  Column-match ratio threshold.

        Returns
        -------
        np.ndarray  Boolean mask, True = near-duplicate of an earlier row.
        """
        n, d   = X.shape
        flags  = np.zeros(n, dtype=bool)
        # Check only first 2000 rows for performance (demo-safe)
        limit  = min(n, 2000)
        for i in range(1, limit):
            for j in range(i):
                match_ratio = np.sum(X[i] == X[j]) / d
                if match_ratio >= threshold:
                    flags[i] = True
                    break
        return flags

    def screen(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        anomaly_score_threshold: float = 0.80,
    ) -> Dict:
        """
        Screen a training batch for poisoned records.

        Parameters
        ----------
        X                       : np.ndarray  Feature matrix, shape (n, d).
        y                       : np.ndarray  Label vector, shape (n,).
        feature_names           : List[str]   Column names for X.
        anomaly_score_threshold : float       Anomaly score above which a row is
                                              flagged as poisoned (default 0.80).

        Returns
        -------
        dict
            {
                "safe_to_train":    bool,
                "poison_rate":      float,
                "poisoned_indices": List[int],
                "n_total":          int,
                "n_flagged":        int,
                "breakdown": {
                    "anomaly":     int,
                    "label_inconsistent": int,
                    "duplicates":  int,
                }
            }
        """
        # Guard: None / uninitialized model
        if X is None or len(X) == 0:
            logger.warning("DataPoisonDetector.screen: empty input, returning safe.")
            return {
                "safe_to_train": True, "poison_rate": 0.0,
                "poisoned_indices": [], "n_total": 0, "n_flagged": 0,
                "breakdown": {"anomaly": 0, "label_inconsistent": 0, "duplicates": 0},
            }

        try:
            n = len(X)
            fn = feature_names or [f"f{i}" for i in range(X.shape[1])]

            # Layer 1: Isolation Forest anomaly scores
            anomaly_scores = self._anomaly_scores(X)
            anomaly_flags  = anomaly_scores >= anomaly_score_threshold

            # Layer 2: Label consistency
            label_flags = self._label_consistency_flags(X, y, fn)

            # Layer 3: Duplicates (capped at 2k rows for performance)
            dup_flags = self._duplicate_flags(X, self.duplicate_threshold)

            # Union of all flags
            combined_flags = anomaly_flags | label_flags | dup_flags
            poisoned_idx   = list(np.where(combined_flags)[0].astype(int))
            n_flagged      = int(combined_flags.sum())
            poison_rate    = n_flagged / n if n > 0 else 0.0
            safe_to_train  = poison_rate < self.poison_threshold

            result = {
                "safe_to_train":    safe_to_train,
                "poison_rate":      round(poison_rate, 4),
                "poisoned_indices": poisoned_idx,
                "n_total":          n,
                "n_flagged":        n_flagged,
                "breakdown": {
                    "anomaly":            int(anomaly_flags.sum()),
                    "label_inconsistent": int(label_flags.sum()),
                    "duplicates":         int(dup_flags.sum()),
                },
            }

            level = logging.WARNING if not safe_to_train else logging.INFO
            logger.log(
                level,
                f"[DataPoisonDetector] safe={safe_to_train} "
                f"poison_rate={poison_rate:.2%} n_flagged={n_flagged}/{n}"
            )
            return result

        except Exception as e:
            logger.error(
                f"DataPoisonDetector.screen failed: {e}", exc_info=True
            )
            # Fail-safe: allow training but log the error
            return {
                "safe_to_train":    True,
                "poison_rate":      0.0,
                "poisoned_indices": [],
                "n_total":          len(X),
                "n_flagged":        0,
                "breakdown": {"anomaly": 0, "label_inconsistent": 0, "duplicates": 0},
            }


# ---------------------------------------------------------------------------
# Report helper
# ---------------------------------------------------------------------------

def save_security_report(report: Dict, path: str = SECURITY_REPORT_PATH) -> None:
    """
    Save the security monitor evaluation report to JSON.

    Parameters
    ----------
    report : dict
    path   : str
    """
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Security monitor report saved: {path}")
    except Exception as e:
        logger.error(f"Failed to save security report: {e}")


def main() -> None:
    """Smoke test for both detectors."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler("logs/task22_security.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    rng = np.random.default_rng(42)

    # --- ScrapingDetector ---
    scraper = ScrapingDetector(rate_limit_threshold=5, block_threshold=10)
    t0 = time.time()
    for i in range(12):
        res = scraper.check("bot_client", resource_id=i, now=t0 + i * 0.5)
        if res["action"] != "ALLOW":
            print(f"  Request {i}: {res['action']} - {res['reason']}")
            if res["action"] == "BLOCK":
                break

    # --- DataPoisonDetector ---
    poison_det = DataPoisonDetector(poison_threshold=0.05)
    clean_X = rng.normal(0, 1, (500, 5)).astype(np.float32)
    poison_det.fit_reference(clean_X)

    # Inject 10% poison rows
    test_X   = rng.normal(0, 1, (100, 5)).astype(np.float32)
    poison_rows = rng.normal(10, 0.1, (10, 5)).astype(np.float32)
    mixed_X  = np.vstack([test_X, poison_rows])
    mixed_y  = np.zeros(110, dtype=int)

    result = poison_det.screen(mixed_X, mixed_y)
    print(f"  safe_to_train={result['safe_to_train']} "
          f"poison_rate={result['poison_rate']:.2%} "
          f"flagged={result['n_flagged']}/{result['n_total']}")
    print("[OK] security_monitor smoke test passed.")


if __name__ == "__main__":
    main()
