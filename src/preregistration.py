"""
preregistration.py -- PlaceMux Phase 3, Task 10
================================================
Pre-registration of experiment hypotheses and metrics.

Anti-gaming contract
--------------------
The hypothesis, primary metric, MDE, alpha, and decision rules are ALL
written and locked BEFORE any experiment data is generated.  The file is
sealed with a SHA-256 content hash.  The readout verifies the hash before
accepting any decision -- if the pre-registration was tampered with after
data collection, the hash check fails and the experiment is invalidated.

This enforces:
- No metric swapping after seeing results
- No threshold moving to claim significance
- No cherry-picking guardrails post-hoc
"""

import os
import sys
import json
import math
import hashlib
import logging
from dataclasses import dataclass, asdict
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
        logging.FileHandler(os.path.join("logs", "task10.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

PREREG_PATH = "logs/preregistration.json"


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class PreRegistration:
    """
    Locked experiment pre-registration document.

    All fields are set at registration time and must not change.
    The content_hash field is computed from all other fields and acts
    as a tamper-evident seal.

    Parameters
    ----------
    experiment_id : str
        Unique experiment identifier (matches ExperimentEngine config).
    hypothesis : str
        One-sentence testable hypothesis, written in plain English.
    primary_metric : str
        Name of the primary decision metric (e.g. 'CTR').
    direction : str
        Expected direction of effect ('increase' or 'decrease').
    baseline_rate : float
        Observed historical baseline rate for the primary metric.
    mde_relative : float
        Minimum Detectable Effect as a relative lift (e.g. 0.05 = +5%).
    alpha : float
        Type I error rate (significance level). Typically 0.05.
    power : float
        Target statistical power (1 - beta). Typically 0.80.
    required_n_per_variant : int
        Pre-computed minimum sample size per variant (from power analysis).
    guardrail_metrics : dict
        Metric -> threshold dict. Experiment halts if any is breached.
    decision_rule : str
        Plain-English decision rule (written before data collection).
    registered_at : str
        UTC ISO timestamp when the registration was written.
    content_hash : str
        SHA-256 hash of all fields (excluding content_hash itself).
    """
    experiment_id: str
    hypothesis: str
    primary_metric: str
    direction: str
    baseline_rate: float
    mde_relative: float
    alpha: float
    power: float
    required_n_per_variant: int
    guardrail_metrics: Dict
    decision_rule: str
    registered_at: str
    content_hash: str = ""


# ---------------------------------------------------------------------------
# Power analysis
# ---------------------------------------------------------------------------

def compute_required_n(
    baseline_rate: float,
    mde_relative: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """
    Compute the minimum sample size per variant for a two-proportion z-test.

    Uses the arcsine transformation (Cohen's h) which is more accurate for
    proportions than the simple normal approximation, especially at extremes.

    Parameters
    ----------
    baseline_rate : float
        Baseline conversion rate (e.g. 0.10 for 10% CTR).
    mde_relative : float
        Minimum detectable effect as relative lift (e.g. 0.05 = +5%).
    alpha : float
        Significance level (one-sided). Default 0.05.
    power : float
        Target power (1 - beta). Default 0.80.

    Returns
    -------
    int
        Required sample size per variant (rounded up to nearest 100).
    """
    treatment_rate = baseline_rate * (1 + mde_relative)

    # Cohen's h effect size for proportions
    h = 2 * (math.asin(math.sqrt(treatment_rate)) - math.asin(math.sqrt(baseline_rate)))

    # Z-scores for alpha and beta
    # z_alpha: one-sided critical value at alpha
    # z_beta: critical value at (1 - power)
    # Using math.erf to invert the normal CDF
    def z_from_p(p: float) -> float:
        """Approximate inverse normal CDF via bisection on math.erf."""
        lo, hi = -6.0, 6.0
        for _ in range(60):
            mid = (lo + hi) / 2
            cdf_mid = 0.5 * (1 + math.erf(mid / math.sqrt(2)))
            if cdf_mid < p:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    z_alpha = z_from_p(1 - alpha)    # e.g. 1.645 for alpha=0.05, one-sided
    z_beta = z_from_p(power)          # e.g. 0.842 for power=0.80

    n = ((z_alpha + z_beta) / h) ** 2

    # Round up to nearest 100 for practical planning
    return int(math.ceil(n / 100) * 100)


# ---------------------------------------------------------------------------
# Registration and verification
# ---------------------------------------------------------------------------

def _compute_hash(prereg_dict: dict) -> str:
    """
    Compute a SHA-256 hash of the pre-registration content.

    Excludes the 'content_hash' field itself from the computation.

    Parameters
    ----------
    prereg_dict : dict
        Pre-registration as a dictionary (content_hash excluded).

    Returns
    -------
    str
        Hex-encoded SHA-256 digest.
    """
    d = {k: v for k, v in prereg_dict.items() if k != "content_hash"}
    serialised = json.dumps(d, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def register(
    experiment_id: str,
    hypothesis: str,
    primary_metric: str,
    direction: str,
    baseline_rate: float,
    mde_relative: float,
    alpha: float = 0.05,
    power: float = 0.80,
    guardrail_metrics: Optional[Dict] = None,
    decision_rule: Optional[str] = None,
    path: str = PREREG_PATH,
) -> PreRegistration:
    """
    Create and lock a pre-registration document.

    Must be called BEFORE any experiment data is generated.
    Writes the pre-registration to disk with a tamper-evident content hash.

    Parameters
    ----------
    experiment_id : str
    hypothesis : str
        One-sentence testable hypothesis.
    primary_metric : str
        Name of the primary decision metric.
    direction : str
        'increase' or 'decrease'.
    baseline_rate : float
        Historical baseline rate for the primary metric.
    mde_relative : float
        Minimum Detectable Effect (relative, e.g. 0.05).
    alpha : float
        Significance level.
    power : float
        Target statistical power.
    guardrail_metrics : dict, optional
        Guardrail thresholds dict.
    decision_rule : str, optional
        Plain-English decision rule.
    path : str
        Output path for the pre-registration JSON.

    Returns
    -------
    PreRegistration
        The locked pre-registration object.
    """
    if guardrail_metrics is None:
        guardrail_metrics = {
            "CTR_floor_relative_to_control": -0.20,
            "apply_rate_floor_relative_to_control": -0.15,
            "max_error_rate": 0.05,
        }

    if decision_rule is None:
        decision_rule = (
            f"SHIP if: p < {alpha} AND relative_lift > {mde_relative:.0%} "
            f"AND all guardrails GREEN. "
            f"DO-NOT-SHIP if: p >= {alpha} OR lift <= {mde_relative:.0%} "
            f"OR any guardrail RED. "
            f"INCONCLUSIVE if: n_per_variant < required_n."
        )

    required_n = compute_required_n(baseline_rate, mde_relative, alpha, power)

    prereg = PreRegistration(
        experiment_id=experiment_id,
        hypothesis=hypothesis,
        primary_metric=primary_metric,
        direction=direction,
        baseline_rate=baseline_rate,
        mde_relative=mde_relative,
        alpha=alpha,
        power=power,
        required_n_per_variant=required_n,
        guardrail_metrics=guardrail_metrics,
        decision_rule=decision_rule,
        registered_at=datetime.utcnow().isoformat(),
        content_hash="",
    )

    # Compute and seal the content hash
    d = asdict(prereg)
    prereg.content_hash = _compute_hash(d)

    # Persist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(prereg), f, indent=2)

    logger.info(
        f"[PREREG] Registered experiment '{experiment_id}' -> {path}"
    )
    logger.info(
        f"[PREREG] Primary metric: {primary_metric} ({direction}), "
        f"MDE={mde_relative:.0%}, alpha={alpha}, power={power}, "
        f"required_n={required_n}/variant"
    )
    logger.info(f"[PREREG] Content hash: {prereg.content_hash[:16]}...")

    return prereg


def load_and_verify(path: str = PREREG_PATH) -> PreRegistration:
    """
    Load a pre-registration from disk and verify its tamper-evident hash.

    If the hash does not match (i.e. the pre-registration was modified after
    creation), raises a ValueError -- the experiment cannot proceed.

    Parameters
    ----------
    path : str
        Path to the pre-registration JSON file.

    Returns
    -------
    PreRegistration
        Verified pre-registration object.

    Raises
    ------
    FileNotFoundError
        If the pre-registration file does not exist.
    ValueError
        If the content hash does not match (tamper detected).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Pre-registration not found at '{path}'. "
            f"Register the experiment before running it."
        )

    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)

    stored_hash = d.get("content_hash", "")
    expected_hash = _compute_hash(d)

    if stored_hash != expected_hash:
        raise ValueError(
            f"[TAMPER DETECTED] Pre-registration hash mismatch!\n"
            f"  Stored  : {stored_hash[:32]}...\n"
            f"  Expected: {expected_hash[:32]}...\n"
            f"  The pre-registration was modified after locking. "
            f"Experiment invalidated."
        )

    prereg = PreRegistration(**d)
    logger.info(
        f"[PREREG] Verified '{prereg.experiment_id}' -- hash OK ({stored_hash[:16]}...)"
    )
    return prereg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Create a default pre-registration for smoke testing."""
    try:
        prereg = register(
            experiment_id="exp_placemux_001",
            hypothesis=(
                "Serving candidates with the v2.0 LightGBM ranker (trained on "
                "recency-weighted features) will increase 7-day CTR by at least "
                "+5% relative to the v1.0 baseline ranker."
            ),
            primary_metric="CTR",
            direction="increase",
            baseline_rate=0.10,
            mde_relative=0.05,
        )
        verified = load_and_verify()
        print(f"\n  Experiment  : {verified.experiment_id}")
        print(f"  Hypothesis  : {verified.hypothesis[:80]}...")
        print(f"  Required n  : {verified.required_n_per_variant} / variant")
        print(f"  Hash OK     : {verified.content_hash[:24]}...\n")
    except Exception as e:
        logger.critical(f"Pre-registration failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
