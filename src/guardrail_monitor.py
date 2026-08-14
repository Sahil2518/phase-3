"""
guardrail_monitor.py -- PlaceMux Phase 3, Task 9
=================================================
Guardrail metrics that automatically halt a bad model experiment.

Design rationale
----------------
Guardrails protect production from a bad model being ramped before the
team notices.  They are NOT the primary success metric -- they are the
safety net.  A guardrail breach means the new model is actively WORSE
on a metric we cannot afford to degrade (hiring relevance, CTR, apply rate).

Guardrail thresholds (configurable, these are the defaults)
-----------------------------------------------------------
  CTR drop       : halt if treatment CTR < control CTR * (1 - 0.20)
                   i.e. treatment cannot lose more than 20% relative CTR
  Apply rate drop: halt if treatment apply_rate < control * (1 - 0.15)
  Error rate      : halt if error_rate > 0.05 (5%)

Statistical gate
----------------
Each guardrail is tested with a one-sided two-proportion z-test.
A guardrail is only breached if:
  (a) the observed metric drop exceeds the threshold, AND
  (b) the drop is statistically significant at p < 0.05

This prevents false alarms from small-sample noise early in an experiment.

Output
------
- logs/guardrail_report.json  (latest check result)
- logs/task09.log             (append-only experiment log)
"""

import os
import sys
import json
import logging
import math
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional

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

GUARDRAIL_REPORT_PATH = "logs/guardrail_report.json"


# ---------------------------------------------------------------------------
# Statistical helper
# ---------------------------------------------------------------------------

def two_proportion_z_test(
    successes_a: int,
    trials_a: int,
    successes_b: int,
    trials_b: int,
) -> float:
    """
    One-sided two-proportion z-test: is proportion_b < proportion_a?

    Returns the p-value for H0: p_b >= p_a (one-sided lower tail).
    A small p-value (< 0.05) means treatment is significantly worse than control.

    Uses the error function (math.erf) for an exact normal CDF -- no scipy needed.

    Parameters
    ----------
    successes_a : int
        Number of successes (clicks/applies) in group A (control).
    trials_a : int
        Number of trials (impressions) in group A.
    successes_b : int
        Number of successes in group B (treatment).
    trials_b : int
        Number of trials in group B.

    Returns
    -------
    float
        p-value for B < A (one-sided). Returns 1.0 if inputs are invalid.
    """
    if trials_a == 0 or trials_b == 0:
        return 1.0  # Cannot determine significance -- conservative

    p_a = successes_a / trials_a
    p_b = successes_b / trials_b
    p_pool = (successes_a + successes_b) / (trials_a + trials_b)

    se = math.sqrt(p_pool * (1 - p_pool) * (1 / trials_a + 1 / trials_b))
    if se == 0:
        return 1.0

    # z < 0 means treatment is worse than control
    z = (p_b - p_a) / se

    # Normal CDF using math.erf (accurate, stdlib only)
    # P(Z <= z) = 0.5 * (1 + erf(z / sqrt(2)))
    p_value = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    return float(p_value)


# ---------------------------------------------------------------------------
# Guardrail Monitor
# ---------------------------------------------------------------------------

class GuardrailMonitor:
    """
    Monitors live experiment metrics and halts bad models automatically.

    The monitor checks three guardrail metrics on every call to .check():
    1. CTR (click-through rate) -- primary guardrail
    2. Apply rate              -- secondary guardrail
    3. Error rate              -- system health guardrail

    If any guardrail is breached (drop exceeds threshold AND is statistically
    significant), the experiment is flagged as HALTED and the engine is
    notified to route all traffic back to control.
    """

    def __init__(
        self,
        ctr_max_relative_drop: float = 0.20,
        apply_rate_max_relative_drop: float = 0.15,
        max_error_rate: float = 0.05,
        significance_level: float = 0.05,
        report_path: str = GUARDRAIL_REPORT_PATH,
    ) -> None:
        """
        Initialise the GuardrailMonitor with configurable thresholds.

        Parameters
        ----------
        ctr_max_relative_drop : float
            Maximum tolerable relative CTR drop (treatment vs control).
            Default 0.20 = treatment CTR must be > 80% of control CTR.
        apply_rate_max_relative_drop : float
            Maximum tolerable relative apply rate drop.
            Default 0.15 = treatment apply rate must be > 85% of control.
        max_error_rate : float
            Maximum tolerable error rate in treatment.
            Default 0.05 = 5%.
        significance_level : float
            p-value threshold for statistical significance.
            Default 0.05.
        report_path : str
            Path to write the guardrail JSON report.
        """
        self.ctr_max_drop = ctr_max_relative_drop
        self.apply_rate_max_drop = apply_rate_max_relative_drop
        self.max_error_rate = max_error_rate
        self.alpha = significance_level
        self.report_path = report_path

        logger.info(
            f"GuardrailMonitor initialised: "
            f"ctr_max_drop={ctr_max_relative_drop:.0%}, "
            f"apply_max_drop={apply_rate_max_relative_drop:.0%}, "
            f"max_error_rate={max_error_rate:.0%}, "
            f"alpha={significance_level}"
        )

    def check(
        self,
        control_impressions: int,
        control_clicks: int,
        control_applies: int,
        treatment_impressions: int,
        treatment_clicks: int,
        treatment_applies: int,
        treatment_errors: int = 0,
        engine=None,  # Optional[ExperimentEngine]
    ) -> dict:
        """
        Run all guardrail checks and halt the experiment if any is breached.

        Parameters
        ----------
        control_impressions : int
        control_clicks : int
        control_applies : int
        treatment_impressions : int
        treatment_clicks : int
        treatment_applies : int
        treatment_errors : int
            Number of error responses from the treatment model.
        engine : ExperimentEngine or None
            If provided and a guardrail is breached, engine.halt() is called.

        Returns
        -------
        dict
            Guardrail check report with status, breaches, and all metrics.
        """
        timestamp = datetime.utcnow().isoformat()

        # Compute rates
        ctrl_ctr = control_clicks / max(control_impressions, 1)
        trt_ctr = treatment_clicks / max(treatment_impressions, 1)
        ctrl_apply = control_applies / max(control_impressions, 1)
        trt_apply = treatment_applies / max(treatment_impressions, 1)
        trt_error_rate = treatment_errors / max(treatment_impressions, 1)

        breaches = []

        # -- Guardrail 1: CTR
        breach_1 = self._check_metric(
            metric_name="CTR",
            control_val=ctrl_ctr,
            treatment_val=trt_ctr,
            control_successes=control_clicks,
            control_trials=control_impressions,
            treatment_successes=treatment_clicks,
            treatment_trials=treatment_impressions,
            max_relative_drop=self.ctr_max_drop,
        )
        if breach_1:
            breaches.append(breach_1)

        # -- Guardrail 2: Apply rate
        breach_2 = self._check_metric(
            metric_name="Apply Rate",
            control_val=ctrl_apply,
            treatment_val=trt_apply,
            control_successes=control_applies,
            control_trials=control_impressions,
            treatment_successes=treatment_applies,
            treatment_trials=treatment_impressions,
            max_relative_drop=self.apply_rate_max_drop,
        )
        if breach_2:
            breaches.append(breach_2)

        # -- Guardrail 3: Error rate (no statistical test needed -- absolute threshold)
        if trt_error_rate > self.max_error_rate:
            breach_3 = {
                "metric": "Error Rate",
                "control_value": 0.0,
                "treatment_value": round(trt_error_rate, 4),
                "threshold": self.max_error_rate,
                "relative_drop": None,
                "p_value": None,
                "reason": (
                    f"Treatment error rate {trt_error_rate:.2%} exceeds "
                    f"max threshold {self.max_error_rate:.0%}"
                ),
            }
            breaches.append(breach_3)

        halted = len(breaches) > 0
        status = "HALTED" if halted else "OK"

        # -- Auto-halt the engine if breached
        if halted and engine is not None:
            reason = "; ".join(b["reason"] for b in breaches)
            engine.halt(reason=reason)

        report = {
            "run_timestamp": timestamp,
            "status": status,
            "experiment_halted": halted,
            "metrics": {
                "control": {
                    "impressions": control_impressions,
                    "ctr": round(ctrl_ctr, 4),
                    "apply_rate": round(ctrl_apply, 4),
                },
                "treatment": {
                    "impressions": treatment_impressions,
                    "ctr": round(trt_ctr, 4),
                    "apply_rate": round(trt_apply, 4),
                    "error_rate": round(trt_error_rate, 4),
                },
            },
            "guardrail_breaches": breaches,
        }

        level = logging.WARNING if halted else logging.INFO
        logger.log(
            level,
            f"[GUARDRAIL] status={status} | "
            f"ctrl_ctr={ctrl_ctr:.4f} trt_ctr={trt_ctr:.4f} | "
            f"ctrl_apply={ctrl_apply:.4f} trt_apply={trt_apply:.4f} | "
            f"breaches={len(breaches)}"
        )

        # Persist report
        try:
            with open(self.report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save guardrail report: {e}")

        return report

    def _check_metric(
        self,
        metric_name: str,
        control_val: float,
        treatment_val: float,
        control_successes: int,
        control_trials: int,
        treatment_successes: int,
        treatment_trials: int,
        max_relative_drop: float,
    ) -> Optional[dict]:
        """
        Check a single metric for a guardrail breach.

        Returns a breach dict if breached, or None if safe.

        Parameters
        ----------
        metric_name : str
        control_val : float
        treatment_val : float
        control_successes : int
        control_trials : int
        treatment_successes : int
        treatment_trials : int
        max_relative_drop : float

        Returns
        -------
        dict or None
        """
        if control_val == 0:
            return None  # Cannot compute relative drop -- skip

        relative_drop = (control_val - treatment_val) / control_val

        # Only breach if: (a) drop exceeds threshold AND (b) significant
        if relative_drop > max_relative_drop:
            p_value = two_proportion_z_test(
                control_successes, control_trials,
                treatment_successes, treatment_trials,
            )
            if p_value < self.alpha:
                return {
                    "metric": metric_name,
                    "control_value": round(control_val, 4),
                    "treatment_value": round(treatment_val, 4),
                    "relative_drop": round(relative_drop, 4),
                    "threshold": max_relative_drop,
                    "p_value": round(p_value, 4),
                    "reason": (
                        f"{metric_name} dropped {relative_drop:.1%} relative to control "
                        f"(threshold: {max_relative_drop:.0%}, p={p_value:.4f})"
                    ),
                }

        return None


# ---------------------------------------------------------------------------
# Main (smoke test)
# ---------------------------------------------------------------------------

def main() -> None:
    """Smoke test: simulate a bad treatment and verify guardrail fires."""
    try:
        monitor = GuardrailMonitor()

        # Good model scenario
        print("\n--- Good Model (no breach expected) ---")
        report = monitor.check(
            control_impressions=5000, control_clicks=500, control_applies=150,
            treatment_impressions=600, treatment_clicks=65, treatment_applies=19,
        )
        print(f"  Status: {report['status']}  |  Breaches: {len(report['guardrail_breaches'])}")

        # Bad model scenario
        print("\n--- Bad Model (guardrail should fire) ---")
        report = monitor.check(
            control_impressions=5000, control_clicks=500, control_applies=150,
            treatment_impressions=600, treatment_clicks=180, treatment_applies=20,
            # Inverted: treatment clicks are high but applies are low -- but also
            # test CTR drop:
        )
        # Simulated hard CTR drop
        report = monitor.check(
            control_impressions=5000, control_clicks=500, control_applies=150,
            treatment_impressions=600, treatment_clicks=240, treatment_applies=10,
        )
        print(f"  Status: {report['status']}  |  Breaches: {len(report['guardrail_breaches'])}")
        for b in report["guardrail_breaches"]:
            print(f"    [BREACH] {b['reason']}")

    except Exception as e:
        logger.critical(f"Smoke test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
