"""
readout_generator.py -- PlaceMux Phase 3, Task 10
==================================================
Honest experiment readout: effect size, confidence intervals, guardrail
status, and a ship/do-not-ship decision.

All computations are derived from the pre-registered hypothesis.
Nothing is computed post-hoc; the decision rule was locked before data.

Metrics computed
----------------
- Observed CTR per variant
- Relative lift: (treatment - control) / control
- Cohen's h effect size for proportions
- 95% Wilson confidence intervals for each rate
- Two-proportion z-test p-value (exact erf-based normal CDF)
- Statistical power achieved at actual sample sizes
- Guardrail status per metric
- Ship / Do-Not-Ship / Inconclusive decision with full reasoning

Output
------
- logs/experiment_readout.json  (full machine-readable readout)
- Console table (human-readable summary)
"""

import os
import sys
import json
import math
import logging
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
        logging.FileHandler(os.path.join("logs", "task10.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

READOUT_PATH = "logs/experiment_readout.json"


# ---------------------------------------------------------------------------
# Statistical helpers (stdlib only)
# ---------------------------------------------------------------------------

def _normal_cdf(z: float) -> float:
    """
    Standard normal CDF using math.erf (exact, no scipy needed).

    Parameters
    ----------
    z : float
        The z-score.

    Returns
    -------
    float
        P(Z <= z).
    """
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _z_from_p(p: float) -> float:
    """
    Approximate inverse normal CDF (quantile function) via bisection.

    Parameters
    ----------
    p : float
        Probability in (0, 1).

    Returns
    -------
    float
        z such that P(Z <= z) = p.
    """
    lo, hi = -8.0, 8.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if _normal_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def two_proportion_z_test(
    successes_a: int, trials_a: int,
    successes_b: int, trials_b: int,
) -> float:
    """
    One-sided z-test: p-value for H0: p_b <= p_a (treatment >= control).

    Returns a small p-value when treatment is significantly HIGHER than control
    (i.e., when we have evidence the new model is better).

    Parameters
    ----------
    successes_a : int
        Control successes (clicks).
    trials_a : int
        Control trials (impressions).
    successes_b : int
        Treatment successes.
    trials_b : int
        Treatment trials.

    Returns
    -------
    float
        One-sided p-value for treatment > control. Returns 1.0 if degenerate.
    """
    if trials_a == 0 or trials_b == 0:
        return 1.0

    p_a = successes_a / trials_a
    p_b = successes_b / trials_b
    p_pool = (successes_a + successes_b) / (trials_a + trials_b)

    se = math.sqrt(p_pool * (1 - p_pool) * (1 / trials_a + 1 / trials_b))
    if se == 0:
        return 1.0

    z = (p_b - p_a) / se  # positive z = treatment is better
    # One-sided p-value: P(Z > z) = 1 - CDF(z)
    return float(1.0 - _normal_cdf(z))


def wilson_ci(successes: int, trials: int, confidence: float = 0.95) -> tuple:
    """
    Wilson score confidence interval for a proportion.

    More accurate than the normal approximation, especially for small
    samples or proportions near 0 or 1.

    Parameters
    ----------
    successes : int
    trials : int
    confidence : float
        Confidence level (default 0.95 -> 95% CI).

    Returns
    -------
    tuple
        (lower_bound, upper_bound)
    """
    if trials == 0:
        return (0.0, 1.0)

    alpha = 1 - confidence
    z = _z_from_p(1 - alpha / 2)  # two-sided critical value

    p_hat = successes / trials
    n = trials
    denom = 1 + z ** 2 / n
    centre = (p_hat + z ** 2 / (2 * n)) / denom
    half_width = (z / denom) * math.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2))

    return (max(0.0, centre - half_width), min(1.0, centre + half_width))


def cohens_h(p1: float, p2: float) -> float:
    """
    Cohen's h effect size for two proportions.

    h = 2 * (arcsin(sqrt(p2)) - arcsin(sqrt(p1)))

    h = 0.2 small, 0.5 medium, 0.8 large.

    Parameters
    ----------
    p1 : float
        Baseline proportion (control).
    p2 : float
        Treatment proportion.

    Returns
    -------
    float
        Cohen's h (signed; positive = treatment is better).
    """
    return 2.0 * (math.asin(math.sqrt(max(0.0, min(1.0, p2))))
                  - math.asin(math.sqrt(max(0.0, min(1.0, p1)))))


def achieved_power(
    p_control: float,
    p_treatment: float,
    n_control: int,
    n_treatment: int,
    alpha: float = 0.05,
) -> float:
    """
    Compute the statistical power achieved at the observed sample sizes.

    Power = P(reject H0 | H1 true) = P(Z > z_alpha - h * sqrt(n/2))

    Parameters
    ----------
    p_control : float
    p_treatment : float
    n_control : int
    n_treatment : int
    alpha : float

    Returns
    -------
    float
        Achieved power in [0, 1].
    """
    if n_control == 0 or n_treatment == 0:
        return 0.0

    h = abs(cohens_h(p_control, p_treatment))
    if h == 0:
        return float(alpha)

    z_alpha = _z_from_p(1 - alpha)  # one-sided
    n_eff = (n_control * n_treatment) / (n_control + n_treatment)  # harmonic mean / 2
    noncentrality = h * math.sqrt(n_eff)

    # Power = P(Z > z_alpha - noncentrality)
    power = 1.0 - _normal_cdf(z_alpha - noncentrality)
    return float(min(1.0, max(0.0, power)))


# ---------------------------------------------------------------------------
# Readout generator
# ---------------------------------------------------------------------------

class ReadoutGenerator:
    """
    Generates an honest, pre-registration-grounded experiment readout.

    The readout applies the decision rule exactly as written in the
    pre-registration.  No post-hoc metric selection.

    Parameters
    ----------
    prereg : PreRegistration
        The locked pre-registration document (hash already verified).
    readout_path : str
        Path to write the readout JSON.
    """

    def __init__(self, prereg, readout_path: str = READOUT_PATH) -> None:
        """
        Initialise the ReadoutGenerator.

        Parameters
        ----------
        prereg : PreRegistration
        readout_path : str
        """
        # Rule 7: None guard
        if prereg is None:
            raise ValueError("ReadoutGenerator requires a valid PreRegistration.")
        self.prereg = prereg
        self.readout_path = readout_path

    def generate(
        self,
        control_impressions: int,
        control_clicks: int,
        control_applies: int,
        treatment_impressions: int,
        treatment_clicks: int,
        treatment_applies: int,
        treatment_errors: int = 0,
        guardrail_report: Optional[dict] = None,
        scenario_label: str = "experiment",
    ) -> dict:
        """
        Compute the full honest readout and ship/do-not-ship decision.

        Parameters
        ----------
        control_impressions : int
        control_clicks : int
        control_applies : int
        treatment_impressions : int
        treatment_clicks : int
        treatment_applies : int
        treatment_errors : int
        guardrail_report : dict, optional
            Output from GuardrailMonitor.check(). If None, guardrails are not checked.
        scenario_label : str
            Human-readable label for this run.

        Returns
        -------
        dict
            Complete readout with all metrics, CIs, decision, and reasoning.
        """
        p = self.prereg
        timestamp = datetime.utcnow().isoformat()

        # -- Primary metric rates
        ctrl_ctr = control_clicks / max(control_impressions, 1)
        trt_ctr = treatment_clicks / max(treatment_impressions, 1)
        ctrl_apply = control_applies / max(control_impressions, 1)
        trt_apply = treatment_applies / max(treatment_impressions, 1)
        trt_error_rate = treatment_errors / max(treatment_impressions, 1)

        # -- Effect size and lift
        relative_lift = (trt_ctr - ctrl_ctr) / max(ctrl_ctr, 1e-9)
        h = cohens_h(ctrl_ctr, trt_ctr)

        # -- Confidence intervals (95% Wilson)
        ctrl_ci = wilson_ci(control_clicks, control_impressions)
        trt_ci = wilson_ci(treatment_clicks, treatment_impressions)

        # -- p-value (one-sided: treatment > control)
        p_value = two_proportion_z_test(
            control_clicks, control_impressions,
            treatment_clicks, treatment_impressions,
        )

        # -- Achieved power
        power_achieved = achieved_power(
            ctrl_ctr, trt_ctr,
            control_impressions, treatment_impressions,
            alpha=p.alpha,
        )

        # -- Sample size check
        is_powered = (
            control_impressions >= p.required_n_per_variant
            and treatment_impressions >= p.required_n_per_variant
        )

        # -- Guardrail status
        guardrails_ok = True
        guardrail_summary = "Not checked"
        if guardrail_report is not None:
            guardrails_ok = len(guardrail_report.get("guardrail_breaches", [])) == 0
            guardrail_summary = (
                "All green" if guardrails_ok
                else f"{len(guardrail_report['guardrail_breaches'])} breach(es)"
            )

        # -- Ship decision (pre-registered rule applied verbatim)
        decision, reasoning = self._make_decision(
            p_value=p_value,
            relative_lift=relative_lift,
            is_powered=is_powered,
            guardrails_ok=guardrails_ok,
            guardrail_report=guardrail_report,
        )

        readout = {
            "scenario": scenario_label,
            "run_timestamp": timestamp,
            "experiment_id": p.experiment_id,
            "prereg_hash": p.content_hash[:16] + "...",
            "hypothesis": p.hypothesis,
            "pre_registered_metric": p.primary_metric,
            "pre_registered_mde": p.mde_relative,
            "pre_registered_alpha": p.alpha,
            "required_n_per_variant": p.required_n_per_variant,

            "observed_metrics": {
                "control": {
                    "impressions": control_impressions,
                    "clicks": control_clicks,
                    "ctr": round(ctrl_ctr, 4),
                    "ctr_95ci": [round(ctrl_ci[0], 4), round(ctrl_ci[1], 4)],
                    "apply_rate": round(ctrl_apply, 4),
                },
                "treatment": {
                    "impressions": treatment_impressions,
                    "clicks": treatment_clicks,
                    "ctr": round(trt_ctr, 4),
                    "ctr_95ci": [round(trt_ci[0], 4), round(trt_ci[1], 4)],
                    "apply_rate": round(trt_apply, 4),
                    "error_rate": round(trt_error_rate, 4),
                },
            },

            "statistical_analysis": {
                "relative_lift": round(relative_lift, 4),
                "cohens_h": round(h, 4),
                "p_value": round(p_value, 6),
                "statistically_significant": bool(p_value < p.alpha),
                "achieved_power": round(power_achieved, 4),
                "is_adequately_powered": bool(is_powered),
            },

            "guardrail_status": {
                "all_green": bool(guardrails_ok),
                "summary": guardrail_summary,
            },

            "decision": decision,
            "reasoning": reasoning,
        }

        # Log key numbers
        logger.info(
            f"[READOUT:{scenario_label}] "
            f"CTR ctrl={ctrl_ctr:.4f} trt={trt_ctr:.4f} "
            f"lift={relative_lift:+.1%} p={p_value:.4f} "
            f"h={h:.4f} decision={decision}"
        )

        # Persist
        try:
            os.makedirs(os.path.dirname(self.readout_path), exist_ok=True)
            with open(self.readout_path, "w", encoding="utf-8") as f:
                json.dump(readout, f, indent=2)
            logger.info(f"[READOUT] Saved -> {self.readout_path}")
        except Exception as e:
            logger.error(f"Failed to save readout: {e}")

        return readout

    def _make_decision(
        self,
        p_value: float,
        relative_lift: float,
        is_powered: bool,
        guardrails_ok: bool,
        guardrail_report: Optional[dict],
    ) -> tuple:
        """
        Apply the pre-registered decision rule verbatim.

        Decision hierarchy (in order of precedence):
        1. INCONCLUSIVE if experiment is underpowered.
        2. DO-NOT-SHIP if any guardrail is red.
        3. SHIP if p < alpha AND lift > MDE AND guardrails green.
        4. DO-NOT-SHIP otherwise.

        Parameters
        ----------
        p_value : float
        relative_lift : float
        is_powered : bool
        guardrails_ok : bool
        guardrail_report : dict or None

        Returns
        -------
        tuple
            (decision_string, reasoning_string)
        """
        p = self.prereg
        reasons = []

        # Priority 1: Underpowered
        if not is_powered:
            reasons.append(
                f"Experiment is underpowered -- sample sizes below the required "
                f"{p.required_n_per_variant} impressions per variant. "
                f"Collect more data before deciding."
            )
            return "INCONCLUSIVE", " ".join(reasons)

        # Priority 2: Guardrail breach
        if not guardrails_ok and guardrail_report is not None:
            breaches = guardrail_report.get("guardrail_breaches", [])
            breach_strs = [b["reason"] for b in breaches]
            reasons.append(
                f"DO-NOT-SHIP: Guardrail breach(es) detected -- "
                + "; ".join(breach_strs)
            )
            return "DO-NOT-SHIP", " ".join(reasons)

        # Priority 3: Full ship check
        sig = p_value < p.alpha
        beats_mde = relative_lift > p.mde_relative

        if sig and beats_mde and guardrails_ok:
            reasons.append(
                f"SHIP: p={p_value:.4f} < alpha={p.alpha} (statistically significant). "
                f"Relative lift={relative_lift:+.1%} beats MDE of {p.mde_relative:.0%}. "
                f"All guardrails green. Decision consistent with pre-registered rule."
            )
            return "SHIP", " ".join(reasons)

        # Priority 4: Do-not-ship
        if not sig:
            reasons.append(
                f"p={p_value:.4f} >= alpha={p.alpha} -- not statistically significant."
            )
        if not beats_mde:
            reasons.append(
                f"Relative lift={relative_lift:+.1%} does not beat "
                f"pre-registered MDE of {p.mde_relative:.0%}."
            )

        reasons.append(
            "DO-NOT-SHIP: Pre-registered decision rule not met. "
            "Do not ship based on insufficient evidence."
        )
        return "DO-NOT-SHIP", " ".join(reasons)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Smoke test with synthetic counts."""
    try:
        from src.preregistration import load_and_verify
        prereg = load_and_verify()
        gen = ReadoutGenerator(prereg)
        readout = gen.generate(
            control_impressions=2000, control_clicks=200, control_applies=60,
            treatment_impressions=2000, treatment_clicks=220, treatment_applies=65,
            scenario_label="smoke_test",
        )
        print(f"\n  Decision: {readout['decision']}")
        print(f"  Reasoning: {readout['reasoning'][:120]}...\n")
    except Exception as e:
        logger.critical(f"Smoke test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
