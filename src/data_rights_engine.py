"""
data_rights_engine.py -- PlaceMux Phase 3, Task 23
===================================================
DPDP / GDPR Data-Subject Rights for the ML Intelligence Layer.

Three rights are implemented:

1. Right of Access (Art. 15 GDPR / DPDP Sec. 11)
   Returns every piece of stored data for a subject_id:
   profile features, model scores, training-set membership.

2. Right to Erasure / 'Right to be Forgotten' (Art. 17 GDPR / DPDP Sec. 13)
   Cascades deletion across:
     - Feature store (in-memory dict simulating a feature DB)
     - Interaction / click log
     - Training queue (prevents future training on this subject)
   Issues a tamper-evident deletion certificate with SHA-256 hash.

3. Retraining Implications (Art. 22 GDPR impact assessment)
   After deletion, assesses whether removing the subject would
   materially change the model (influence score heuristic).
   If influence exceeds a threshold, sets retrain_required=True.

Design rationale
----------------
The feature store and log are in-memory dicts (seeded deterministically)
to keep the demo self-contained.  A production system would wire these
to Redis / BigQuery / PostgreSQL via the same interface.

All requests are appended to logs/data_rights_log.jsonl for audit.
Deletion certificates are written to logs/audit_pack/.

Metric bar
----------
  Access  : returns full data bundle < 1s
  Deletion: all affected records purged, certificate issued < 5s
  Retrain : correctly schedules retrain when influence > threshold
"""

import os
import sys
import json
import hashlib
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

os.makedirs("logs", exist_ok=True)
os.makedirs("logs/audit_pack", exist_ok=True)
logger = logging.getLogger(__name__)

RIGHTS_LOG_PATH = "logs/data_rights_log.jsonl"
CERT_DIR        = "logs/audit_pack"


# ---------------------------------------------------------------------------
# In-memory stores (deterministic seed for reproducibility)
# ---------------------------------------------------------------------------

def _build_feature_store(n: int = 200, seed: int = 42) -> Dict[str, Dict]:
    """
    Build a synthetic feature store keyed by candidate_id.

    Parameters
    ----------
    n    : int  Number of candidate records.
    seed : int  Random seed for reproducibility.

    Returns
    -------
    Dict[str, Dict]  Map of candidate_id -> feature dict.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    store = {}
    for i in range(n):
        cid = f"CAND_{i:04d}"
        store[cid] = {
            "candidate_id":       cid,
            "days_since_login":   int(rng.integers(0, 90)),
            "sessions_14d":       int(rng.integers(0, 20)),
            "apply_rate_7d":      round(float(rng.uniform(0, 0.5)), 4),
            "jobs_viewed":        int(rng.integers(0, 300)),
            "profile_completeness": round(float(rng.uniform(20, 100)), 1),
            "is_verified":        int(rng.integers(0, 2)),
            "seniority":          ["junior", "mid", "senior"][int(rng.integers(0, 3))],
            "stored_at":          "2026-08-01T00:00:00Z",
        }
    return store


def _build_interaction_log(
    candidates: List[str], seed: int = 42
) -> List[Dict]:
    """
    Build a synthetic interaction log (click / apply events).

    Parameters
    ----------
    candidates : List[str]  Candidate IDs from the feature store.
    seed       : int

    Returns
    -------
    List[Dict]  List of interaction event dicts.
    """
    import numpy as np
    rng  = np.random.default_rng(seed)
    events = []
    for cid in candidates:
        n_events = int(rng.integers(0, 6))
        for _ in range(n_events):
            events.append({
                "candidate_id": cid,
                "event_type":   rng.choice(["click", "apply", "view"]),
                "job_id":       f"JOB_{int(rng.integers(1000, 2000))}",
                "timestamp":    "2026-08-10T12:00:00Z",
            })
    return events


def _build_score_log(
    candidates: List[str], seed: int = 42
) -> List[Dict]:
    """
    Build a synthetic model-score log (ranking decisions).

    Parameters
    ----------
    candidates : List[str]
    seed       : int

    Returns
    -------
    List[Dict]
    """
    import numpy as np
    rng    = np.random.default_rng(seed)
    scores = []
    for cid in candidates:
        scores.append({
            "candidate_id":  cid,
            "job_id":        f"JOB_{int(rng.integers(1000, 2000))}",
            "model_version": "churn_model_v2",
            "raw_score":     round(float(rng.uniform(0.2, 0.95)), 4),
            "rank":          int(rng.integers(1, 50)),
            "scored_at":     "2026-08-15T09:00:00Z",
        })
    return scores


def _build_training_queue(candidates: List[str]) -> List[str]:
    """
    Build a synthetic training queue: subset of candidates queued for next retrain.

    Parameters
    ----------
    candidates : List[str]

    Returns
    -------
    List[str]  Candidate IDs in the training queue.
    """
    return candidates[:100]   # first 100 are queued


# ---------------------------------------------------------------------------
# DataSubjectRightsEngine
# ---------------------------------------------------------------------------

class DataSubjectRightsEngine:
    """
    Handles DPDP / GDPR data-subject rights requests for the ML pipeline.

    Parameters
    ----------
    influence_threshold : float  Data-influence score above which a retrain is
                                 scheduled after deletion (default 0.70).
    n_candidates        : int    Number of synthetic candidates to generate.
    seed                : int    Random seed.
    """

    def __init__(
        self,
        influence_threshold: float = 0.70,
        n_candidates:        int   = 200,
        seed:                int   = 42,
    ) -> None:
        """Initialise the engine and build in-memory stores."""
        self.influence_threshold = influence_threshold

        # Build stores
        self._feature_store    = _build_feature_store(n_candidates, seed)
        all_candidates         = list(self._feature_store.keys())
        self._interaction_log  = _build_interaction_log(all_candidates, seed)
        self._score_log        = _build_score_log(all_candidates, seed)
        self._training_queue   = set(_build_training_queue(all_candidates))
        self._deleted_subjects: set = set()

        logger.info(
            f"DataSubjectRightsEngine ready: "
            f"{len(self._feature_store)} candidates, "
            f"{len(self._interaction_log)} interaction events, "
            f"{len(self._score_log)} score records"
        )

    # ------------------------------------------------------------------
    # Right of Access
    # ------------------------------------------------------------------

    def handle_access_request(self, subject_id: str) -> Dict:
        """
        Return every piece of stored data for a subject (Right of Access).

        Parameters
        ----------
        subject_id : str  Candidate ID.

        Returns
        -------
        dict  Full data bundle with features, scores, events, training membership.
        """
        if not subject_id:
            raise ValueError("subject_id must be a non-empty string.")

        features    = self._feature_store.get(subject_id)
        events      = [e for e in self._interaction_log
                       if e["candidate_id"] == subject_id]
        scores      = [s for s in self._score_log
                       if s["candidate_id"] == subject_id]
        in_training = subject_id in self._training_queue

        bundle = {
            "request_type":      "access",
            "subject_id":        subject_id,
            "request_timestamp": datetime.now(timezone.utc).isoformat(),
            "found":             features is not None,
            "profile_features":  features or {},
            "model_scores":      scores,
            "interaction_events": events,
            "in_training_queue": in_training,
            "training_membership_note": (
                "Your data is included in the next model training batch."
                if in_training else
                "Your data is NOT currently in the training queue."
            ),
        }

        self._append_audit_log(bundle)
        logger.info(
            f"[Access] subject={subject_id} "
            f"found={features is not None} "
            f"events={len(events)} scores={len(scores)}"
        )
        return bundle

    # ------------------------------------------------------------------
    # Right to Erasure
    # ------------------------------------------------------------------

    def handle_deletion_request(self, subject_id: str) -> Dict:
        """
        Cascade deletion across all stores and issue a deletion certificate.

        Parameters
        ----------
        subject_id : str  Candidate ID.

        Returns
        -------
        dict  Deletion certificate with counts and SHA-256 hash.
        """
        if not subject_id:
            raise ValueError("subject_id must be a non-empty string.")

        ts = datetime.now(timezone.utc).isoformat()

        # Count before deletion
        had_features   = subject_id in self._feature_store
        events_before  = len([e for e in self._interaction_log
                               if e["candidate_id"] == subject_id])
        scores_before  = len([s for s in self._score_log
                               if s["candidate_id"] == subject_id])
        in_queue_before = subject_id in self._training_queue

        # --- Cascade deletion ---
        # 1. Feature store
        self._feature_store.pop(subject_id, None)

        # 2. Interaction log
        self._interaction_log = [
            e for e in self._interaction_log
            if e["candidate_id"] != subject_id
        ]

        # 3. Score log
        self._score_log = [
            s for s in self._score_log
            if s["candidate_id"] != subject_id
        ]

        # 4. Training queue
        self._training_queue.discard(subject_id)
        self._deleted_subjects.add(subject_id)

        # Issue deletion certificate
        cert_payload = {
            "request_type":         "deletion",
            "subject_id":           subject_id,
            "deletion_timestamp":   ts,
            "records_deleted": {
                "feature_profile":  1 if had_features else 0,
                "interaction_events": events_before,
                "model_scores":     scores_before,
                "training_queue":   1 if in_queue_before else 0,
            },
            "total_records_deleted": (
                (1 if had_features else 0)
                + events_before + scores_before
                + (1 if in_queue_before else 0)
            ),
            "retrain_scheduled": False,
            "status": "COMPLETE",
        }

        # Compute certificate hash for tamper-evidence
        cert_str  = json.dumps(cert_payload, sort_keys=True)
        cert_hash = hashlib.sha256(cert_str.encode()).hexdigest()
        cert_payload["certificate_hash"] = cert_hash

        # Persist certificate
        cert_path = os.path.join(CERT_DIR, f"deletion_cert_{subject_id}.json")
        try:
            with open(cert_path, "w", encoding="utf-8") as f:
                json.dump(cert_payload, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write deletion certificate: {e}")

        self._append_audit_log(cert_payload)
        logger.info(
            f"[Deletion] subject={subject_id} "
            f"total_deleted={cert_payload['total_records_deleted']} "
            f"cert={cert_hash[:16]}..."
        )
        return cert_payload

    # ------------------------------------------------------------------
    # Retraining Implications
    # ------------------------------------------------------------------

    def assess_retraining_impact(
        self, subject_id: str, deletion_cert: Dict
    ) -> Dict:
        """
        Assess whether deleting this subject materially impacts the model.

        Heuristic: subjects with high activity (many events + high apply rate)
        in the top quartile of the training set have disproportionate influence.

        Parameters
        ----------
        subject_id    : str   Already-deleted candidate.
        deletion_cert : dict  Certificate returned by handle_deletion_request.

        Returns
        -------
        dict  Impact assessment with influence score and retrain_required flag.
        """
        # Events deleted is a proxy for training influence
        events_deleted = deletion_cert["records_deleted"]["interaction_events"]
        scores_deleted = deletion_cert["records_deleted"]["model_scores"]

        # Normalise: assume max meaningful events = 10
        influence_score = round(
            min((events_deleted + scores_deleted) / 10.0, 1.0), 4
        )
        retrain_required = influence_score >= self.influence_threshold

        assessment = {
            "subject_id":       subject_id,
            "influence_score":  influence_score,
            "influence_threshold": self.influence_threshold,
            "retrain_required": retrain_required,
            "reason": (
                f"Subject had {events_deleted} interaction events and "
                f"{scores_deleted} score records. "
                f"Influence score {influence_score:.2f} "
                + (">= threshold -> retrain scheduled."
                   if retrain_required else
                   "< threshold -> no immediate retrain needed.")
            ),
        }
        self._append_audit_log({**assessment, "request_type": "retrain_assessment"})
        logger.info(
            f"[RetainAssess] subject={subject_id} "
            f"influence={influence_score:.2f} retrain={retrain_required}"
        )
        return assessment

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _append_audit_log(self, record: Dict) -> None:
        """
        Append a record to the append-only JSONL audit log.

        Parameters
        ----------
        record : dict  Any dict to persist.
        """
        try:
            with open(RIGHTS_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.error(f"Failed to append to audit log: {e}")

    def summary(self) -> Dict:
        """
        Return a summary of the current store state (for the demo).

        Returns
        -------
        dict
        """
        return {
            "candidates_in_feature_store": len(self._feature_store),
            "interaction_events":          len(self._interaction_log),
            "score_records":               len(self._score_log),
            "training_queue_size":         len(self._training_queue),
            "deleted_subjects":            len(self._deleted_subjects),
        }


def main() -> None:
    """Smoke test."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler("logs/task23.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    try:
        engine = DataSubjectRightsEngine()
        print(engine.handle_access_request("CAND_0001"))
        cert = engine.handle_deletion_request("CAND_0001")
        print(f"Deleted {cert['total_records_deleted']} records")
        impact = engine.assess_retraining_impact("CAND_0001", cert)
        print(f"Retrain required: {impact['retrain_required']}")
        print("[OK] data_rights_engine smoke test passed.")
    except Exception as e:
        logger.critical(f"Smoke test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
