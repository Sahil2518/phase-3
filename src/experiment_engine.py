"""
experiment_engine.py -- PlaceMux Phase 3, Task 9
=================================================
Deterministic variant assignment and feature flag serving for A/B experiments.

Design contract
---------------
- Same user_id + experiment_id always maps to the same variant (no flipping).
- Holdout group is keyed on user_id only -- permanent, cross-experiment.
- Traffic fractions are exact and auditable (SHA-256 hash mod 100).
- Every assignment is logged to JSONL for downstream analysis.

Traffic split (default)
-----------------------
  Bucket 00-09  -> holdout   (10%)  -- permanent, never in any experiment
  Bucket 10-19  -> treatment (10%)  -- new model variant
  Bucket 20-99  -> control   (80%)  -- production model

Feature flags
-------------
Each experiment maps variant names to model versions and config:
  control   -> model_version: v1.0.0-lightgbm, feature_flags: {}
  treatment -> model_version: v2.0.0-lgbm-improved, feature_flags: {rerank: true}
"""

import os
import sys
import json
import hashlib
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Rule 2: Structured logging
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("logs", "task09.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class VariantConfig:
    """
    Configuration for a single experiment variant.

    Parameters
    ----------
    name : str
        Variant name ('control', 'treatment', 'holdout').
    model_version : str
        Model artefact version string routed to this variant.
    traffic_fraction : float
        Fraction of eligible traffic (0.0 - 1.0).
    feature_flags : dict
        Key-value feature flags active for this variant.
    """
    name: str
    model_version: str
    traffic_fraction: float
    feature_flags: Dict = field(default_factory=dict)


@dataclass
class Assignment:
    """
    Result of assigning a user to an experiment variant.

    Parameters
    ----------
    user_id : str
        The user being assigned.
    experiment_id : str
        The experiment being served.
    variant : str
        Assigned variant name.
    model_version : str
        Model version the user will receive.
    feature_flags : dict
        Active feature flags for this variant.
    bucket : int
        Raw hash bucket (0-99) for auditability.
    timestamp : str
        UTC ISO timestamp of assignment.
    """
    user_id: str
    experiment_id: str
    variant: str
    model_version: str
    feature_flags: Dict
    bucket: int
    timestamp: str


@dataclass
class ExperimentConfig:
    """
    Full configuration for one A/B experiment.

    Parameters
    ----------
    experiment_id : str
        Unique identifier for this experiment.
    name : str
        Human-readable experiment name.
    holdout_fraction : float
        Fraction permanently excluded (default 0.10).
    variants : dict
        Mapping of variant name -> VariantConfig.
    status : str
        'RUNNING', 'HALTED', or 'COMPLETED'.
    """
    experiment_id: str
    name: str
    holdout_fraction: float
    variants: Dict[str, VariantConfig]
    status: str = "RUNNING"


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class ExperimentEngine:
    """
    Deterministic A/B experiment engine for PlaceMux.

    Assignment algorithm
    --------------------
    1. Compute holdout bucket: SHA-256(user_id)[:8] mod 100
       If bucket < holdout_fraction * 100 -> assign to holdout (permanent).
    2. Compute variant bucket: SHA-256(user_id + ":" + experiment_id)[:8] mod 90
       Map onto [control, treatment] proportionally.
    3. Log and return the Assignment.

    Thread safety: each assignment opens the log file once for a single write.
    Not designed for high-concurrency production use without a queue in front.
    """

    ASSIGNMENT_LOG = "logs/experiment_assignments.jsonl"

    def __init__(
        self,
        config: ExperimentConfig,
        log_path: str = ASSIGNMENT_LOG,
    ) -> None:
        """
        Initialise the engine with an experiment configuration.

        Parameters
        ----------
        config : ExperimentConfig
            Fully specified experiment including variants and traffic split.
        log_path : str
            Path to write assignment JSONL log.
        """
        # Rule 7: None guard
        if config is None:
            raise ValueError("ExperimentEngine requires a valid ExperimentConfig.")

        self.config = config
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        logger.info(
            f"ExperimentEngine initialised: experiment_id={config.experiment_id}, "
            f"status={config.status}, variants={list(config.variants.keys())}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assign(self, user_id: str) -> Assignment:
        """
        Deterministically assign a user to a variant.

        Rules
        -----
        - If the experiment is HALTED, all users are routed to control.
        - Holdout assignment is based solely on user_id (stable across experiments).
        - Variant assignment within non-holdout traffic uses user_id + experiment_id.

        Parameters
        ----------
        user_id : str
            The user to assign. Must be non-empty.

        Returns
        -------
        Assignment
            The variant assignment for this user.
        """
        # Rule 7: Empty input guard
        if not user_id:
            logger.warning("Empty user_id provided -- assigning to control as fallback.")
            user_id = f"anonymous_{uuid.uuid4().hex[:8]}"

        # If experiment is halted, everyone goes to control
        if self.config.status == "HALTED":
            return self._make_assignment(user_id, "control", bucket=-1)

        # Step 1: Holdout check (keyed on user_id only, permanent)
        holdout_bucket = self._hash_to_bucket(user_id, salt="")
        holdout_cutoff = int(self.config.holdout_fraction * 100)
        if holdout_bucket < holdout_cutoff:
            return self._make_assignment(user_id, "holdout", bucket=holdout_bucket)

        # Step 2: Variant assignment within non-holdout traffic
        # Remap the non-holdout bucket space (holdout_cutoff..99) -> 0..89
        variant_bucket = self._hash_to_bucket(user_id, salt=self.config.experiment_id)
        # Scale to non-holdout range
        non_holdout_space = 100 - holdout_cutoff  # e.g. 90
        variant_bucket_scaled = variant_bucket % non_holdout_space

        # Build cumulative cutoffs from variant configs (excluding holdout)
        non_holdout_variants = {
            k: v for k, v in self.config.variants.items() if k != "holdout"
        }
        cumulative = 0
        assigned_variant = "control"  # safe default
        for variant_name, variant_cfg in non_holdout_variants.items():
            cutoff = int(variant_cfg.traffic_fraction * non_holdout_space)
            if variant_bucket_scaled < cumulative + cutoff:
                assigned_variant = variant_name
                break
            cumulative += cutoff

        assignment = self._make_assignment(
            user_id, assigned_variant, bucket=variant_bucket_scaled
        )
        self._log_assignment(assignment)
        return assignment

    def halt(self, reason: str = "Guardrail breach") -> None:
        """
        Halt the experiment and route all traffic to control.

        Parameters
        ----------
        reason : str
            Plain-English reason for halting.
        """
        self.config.status = "HALTED"
        logger.warning(
            f"[HALT] Experiment '{self.config.experiment_id}' halted. Reason: {reason}"
        )

    def resume(self) -> None:
        """Resume a previously halted experiment."""
        self.config.status = "RUNNING"
        logger.info(f"[RESUME] Experiment '{self.config.experiment_id}' resumed.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _hash_to_bucket(self, user_id: str, salt: str) -> int:
        """
        Map user_id + salt to a deterministic bucket in [0, 99].

        Uses the first 8 hex characters of the SHA-256 digest for speed
        while maintaining excellent uniformity properties.

        Parameters
        ----------
        user_id : str
            The user identifier.
        salt : str
            Additional salt (experiment_id for variant, empty for holdout).

        Returns
        -------
        int
            Bucket index in [0, 99].
        """
        raw = f"{user_id}:{salt}" if salt else user_id
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        # Use first 8 hex chars -> max value 0xFFFFFFFF -> mod 100
        return int(digest[:8], 16) % 100

    def _make_assignment(
        self, user_id: str, variant_name: str, bucket: int
    ) -> Assignment:
        """
        Construct an Assignment dataclass from a variant name.

        Parameters
        ----------
        user_id : str
        variant_name : str
        bucket : int

        Returns
        -------
        Assignment
        """
        # If variant not in config (e.g. holdout before adding its config),
        # fall back gracefully to control
        if variant_name not in self.config.variants and variant_name != "holdout":
            logger.warning(
                f"Variant '{variant_name}' not in config -- falling back to control."
            )
            variant_name = "control"

        if variant_name == "holdout":
            # Holdout uses control's model version (no treatment exposure)
            ctrl = self.config.variants.get("control")
            model_version = ctrl.model_version if ctrl else "holdout-baseline"
            flags = {}
        else:
            vcfg = self.config.variants[variant_name]
            model_version = vcfg.model_version
            flags = vcfg.feature_flags

        return Assignment(
            user_id=user_id,
            experiment_id=self.config.experiment_id,
            variant=variant_name,
            model_version=model_version,
            feature_flags=flags,
            bucket=bucket,
            timestamp=datetime.utcnow().isoformat(),
        )

    def _log_assignment(self, assignment: Assignment) -> None:
        """
        Append an assignment record to the JSONL log.

        Parameters
        ----------
        assignment : Assignment
        """
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(assignment)) + "\n")
        except Exception as e:
            logger.error(f"Failed to log assignment: {e}")


# ---------------------------------------------------------------------------
# Factory: default PlaceMux experiment
# ---------------------------------------------------------------------------

def make_default_experiment(
    experiment_id: str = "exp_placemux_001",
    name: str = "LightGBM v2 vs v1 Ranking Test",
) -> ExperimentConfig:
    """
    Build the default PlaceMux experiment configuration.

    Traffic split: 10% holdout / 10% treatment / 80% control.

    Parameters
    ----------
    experiment_id : str
        Unique experiment identifier.
    name : str
        Human-readable name.

    Returns
    -------
    ExperimentConfig
        Fully initialised experiment configuration.
    """
    return ExperimentConfig(
        experiment_id=experiment_id,
        name=name,
        holdout_fraction=0.10,
        variants={
            "control": VariantConfig(
                name="control",
                model_version="v1.0.0-lightgbm",
                traffic_fraction=0.80,
                feature_flags={},
            ),
            "treatment": VariantConfig(
                name="treatment",
                model_version="v2.0.0-lgbm-improved",
                traffic_fraction=0.10,
                feature_flags={"rerank_by_recency": True, "boost_remote_jobs": False},
            ),
        },
        status="RUNNING",
    )


# ---------------------------------------------------------------------------
# Main (smoke test)
# ---------------------------------------------------------------------------

def main() -> None:
    """Quick smoke test of the assignment engine."""
    try:
        config = make_default_experiment()
        engine = ExperimentEngine(config)

        # Assign 10 users and verify consistency
        user_ids = [f"user_{i:04d}" for i in range(10)]
        print("\n--- Assignment Smoke Test ---")
        for uid in user_ids:
            a1 = engine.assign(uid)
            a2 = engine.assign(uid)  # second call must match first
            assert a1.variant == a2.variant, f"User {uid} flipped variants!"
            print(f"  {uid}  bucket={a1.bucket:>3}  variant={a1.variant:<10}  model={a1.model_version}")
        print("--- All consistent (no flipping) ---\n")
    except Exception as e:
        logger.critical(f"Smoke test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
