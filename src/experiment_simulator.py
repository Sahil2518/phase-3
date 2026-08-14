"""
experiment_simulator.py -- PlaceMux Phase 3, Task 9
====================================================
Generates realistic experiment telemetry for control, treatment, and holdout
groups across three model scenarios.

Scenarios
---------
1. GOOD model  : treatment CTR is +12% relative to control -> no halt
2. BAD model   : treatment CTR is -30% relative to control -> guardrail fires
3. NEUTRAL model: treatment CTR is +2% (noise-level) -> no halt, no clear winner

Each scenario writes events to logs/experiment_events.jsonl, then the
guardrail monitor is run to check whether the halt fires correctly.

Event schema
------------
{
  "user_id": "user_XXXX",
  "variant": "control" | "treatment" | "holdout",
  "experiment_id": "exp_placemux_001",
  "event_type": "impression" | "click" | "apply",
  "timestamp": "<UTC ISO>",
  "model_version": "<version string>"
}
"""

import os
import sys
import json
import logging
import numpy as np
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

EVENTS_LOG_PATH = "logs/experiment_events.jsonl"

# Rule 5: reproducibility
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class ExperimentSimulator:
    """
    Simulates experiment traffic for control, treatment, and holdout groups.

    The simulator uses the ExperimentEngine for deterministic user assignment,
    then samples click/apply events from per-variant conversion probabilities.

    Parameters
    ----------
    engine : ExperimentEngine
        Configured experiment engine for user assignment.
    log_path : str
        Path to write experiment events JSONL.
    random_state : int
        Seed for reproducibility.
    """

    def __init__(
        self,
        engine,  # ExperimentEngine
        log_path: str = EVENTS_LOG_PATH,
        random_state: int = RANDOM_STATE,
    ) -> None:
        """
        Initialise the simulator.

        Parameters
        ----------
        engine : ExperimentEngine
            Configured experiment engine.
        log_path : str
            Destination for events JSONL.
        random_state : int
            Numpy random seed.
        """
        # Rule 7: None guard
        if engine is None:
            raise ValueError("ExperimentSimulator requires a valid ExperimentEngine.")

        self.engine = engine
        self.log_path = log_path
        self.rng = np.random.default_rng(random_state)

        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        logger.info(
            f"ExperimentSimulator initialised. "
            f"Log: {log_path}, seed={random_state}"
        )

    def simulate(
        self,
        num_users: int,
        control_ctr: float,
        treatment_ctr_multiplier: float,
        apply_rate_given_click: float = 0.30,
        error_rate_treatment: float = 0.0,
        scenario_label: str = "scenario",
        append: bool = False,
    ) -> dict:
        """
        Simulate experiment traffic and write events to the log.

        For each user:
        1. Assign to a variant via ExperimentEngine.
        2. Log one 'impression' event.
        3. Sample whether the user clicked (variant-specific CTR).
        4. If clicked, sample whether the user applied.

        Parameters
        ----------
        num_users : int
            Number of unique users to simulate.
        control_ctr : float
            Click-through rate for control and holdout groups.
        treatment_ctr_multiplier : float
            Multiplier on control_ctr for treatment group.
            1.12 -> +12% CTR (good model)
            0.70 -> -30% CTR (bad model)
        apply_rate_given_click : float
            Probability of applying given a click (same across variants for fairness).
        error_rate_treatment : float
            Fraction of treatment impressions that result in an error (no score returned).
        scenario_label : str
            Label attached to log for identification.
        append : bool
            If False, clear the log before writing (default: fresh log per scenario).

        Returns
        -------
        dict
            Aggregated counts per variant for downstream guardrail check.
        """
        # Rule 7: Empty input guard
        if num_users <= 0:
            logger.warning("num_users <= 0 -- skipping simulation.")
            return {}

        mode = "a" if append else "w"
        treatment_ctr = control_ctr * treatment_ctr_multiplier
        # Holdout uses control's model but no experiment exposure -- same CTR as control
        holdout_ctr = control_ctr

        counts = {
            "control":   {"impressions": 0, "clicks": 0, "applies": 0, "errors": 0},
            "treatment": {"impressions": 0, "clicks": 0, "applies": 0, "errors": 0},
            "holdout":   {"impressions": 0, "clicks": 0, "applies": 0, "errors": 0},
        }

        logger.info(
            f"[SIM:{scenario_label}] Users={num_users}, "
            f"ctrl_ctr={control_ctr:.3f}, "
            f"trt_ctr={treatment_ctr:.3f} (x{treatment_ctr_multiplier}), "
            f"error_rate={error_rate_treatment:.2%}"
        )

        events_buffer = []

        for i in range(num_users):
            user_id = f"user_{i:05d}"

            # Rule 7: fault isolation -- one bad user must not crash the batch
            try:
                assignment = self.engine.assign(user_id)
                variant = assignment.variant
                model_version = assignment.model_version

                # Select CTR for this variant
                if variant == "treatment":
                    ctr = treatment_ctr
                else:
                    ctr = control_ctr  # control and holdout use same CTR

                # Impression
                counts[variant]["impressions"] += 1
                events_buffer.append(self._make_event(
                    user_id, variant, "impression", assignment.experiment_id, model_version
                ))

                # Error check for treatment
                if variant == "treatment" and self.rng.random() < error_rate_treatment:
                    counts[variant]["errors"] += 1
                    continue  # error -- no click/apply

                # Click
                if self.rng.random() < ctr:
                    counts[variant]["clicks"] += 1
                    events_buffer.append(self._make_event(
                        user_id, variant, "click", assignment.experiment_id, model_version
                    ))

                    # Apply
                    if self.rng.random() < apply_rate_given_click:
                        counts[variant]["applies"] += 1
                        events_buffer.append(self._make_event(
                            user_id, variant, "apply", assignment.experiment_id, model_version
                        ))

            except Exception as e:
                logger.error(f"Error simulating user {user_id}: {e}")

        # Flush events to disk in one write
        try:
            with open(self.log_path, mode, encoding="utf-8") as f:
                for event in events_buffer:
                    f.write(json.dumps(event) + "\n")
            logger.info(
                f"[SIM:{scenario_label}] Written {len(events_buffer)} events to {self.log_path}"
            )
        except Exception as e:
            logger.error(f"Failed to write events: {e}")

        # Log per-variant summary
        for v, c in counts.items():
            imp = c["impressions"]
            if imp > 0:
                ctr_obs = c["clicks"] / imp
                apr_obs = c["applies"] / imp
                logger.info(
                    f"  [{scenario_label}] {v}: "
                    f"impressions={imp}, "
                    f"ctr={ctr_obs:.4f}, "
                    f"apply_rate={apr_obs:.4f}"
                )

        return counts

    def _make_event(
        self,
        user_id: str,
        variant: str,
        event_type: str,
        experiment_id: str,
        model_version: str,
    ) -> dict:
        """
        Build a single event dictionary.

        Parameters
        ----------
        user_id : str
        variant : str
        event_type : str
        experiment_id : str
        model_version : str

        Returns
        -------
        dict
        """
        return {
            "user_id": user_id,
            "variant": variant,
            "experiment_id": experiment_id,
            "event_type": event_type,
            "model_version": model_version,
            "timestamp": datetime.utcnow().isoformat(),
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Smoke test -- run a single good-model scenario."""
    try:
        from src.experiment_engine import make_default_experiment, ExperimentEngine
        config = make_default_experiment()
        engine = ExperimentEngine(config)
        simulator = ExperimentSimulator(engine)
        counts = simulator.simulate(
            num_users=1000,
            control_ctr=0.10,
            treatment_ctr_multiplier=1.12,
            scenario_label="smoke_test",
        )
        print(f"\nSimulation counts: {counts}\n")
    except Exception as e:
        logger.critical(f"Smoke test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
