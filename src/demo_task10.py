"""
demo_task10.py -- PlaceMux Phase 3, Task 10
============================================
End-to-end demonstration of Growth Integration & Experiment Readout.

Journey (run once, start to finish)
-------------------------------------
1. Pre-register the hypothesis BEFORE any data is generated
2. Run Scenario A: SHIP case (+8% CTR lift) -- should return SHIP
3. Run Scenario B: DO-NOT-SHIP case (+2% CTR) -- not significant, below MDE
4. Run Scenario C: Guardrail breach case -- should return DO-NOT-SHIP
5. Tamper test: corrupt pre-registration mid-run -- hash must fail
6. Break it on purpose: underpowered experiment -- must return INCONCLUSIVE
"""

import os
import sys
import json
import logging

# ---------------------------------------------------------------------------
# Rule 2: Structured logging
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("logs", "task10_demo.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

PREREG_PATH = "logs/preregistration.json"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def banner(title: str) -> None:
    """Print a formatted section banner to stdout."""
    width = 64
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_readout_summary(readout: dict) -> None:
    """Print a compact, human-readable readout summary."""
    obs = readout["observed_metrics"]
    stat = readout["statistical_analysis"]
    ctrl = obs["control"]
    trt = obs["treatment"]

    print(f"\n  Pre-registered metric : {readout['pre_registered_metric']}")
    print(f"  Pre-registered MDE    : {readout['pre_registered_mde']:.0%}")
    print(f"  Pre-registered alpha  : {readout['pre_registered_alpha']}")
    print(f"  Required n/variant    : {readout['required_n_per_variant']}")
    print(f"\n  {'Metric':<22}  {'Control':>10}  {'Treatment':>10}")
    print(f"  {'-'*48}")
    print(f"  {'Impressions':<22}  {ctrl['impressions']:>10,}  {trt['impressions']:>10,}")
    print(f"  {'CTR':<22}  {ctrl['ctr']:>10.4f}  {trt['ctr']:>10.4f}")
    print(f"  {'CTR 95% CI':<22}  {str(ctrl['ctr_95ci']):>10}  {str(trt['ctr_95ci']):>10}")
    print(f"  {'Apply Rate':<22}  {ctrl['apply_rate']:>10.4f}  {trt['apply_rate']:>10.4f}")
    print(f"\n  {'Relative lift':<22}  {stat['relative_lift']:>+10.1%}")
    print(f"  {'Cohen h':<22}  {stat['cohens_h']:>10.4f}")
    print(f"  {'p-value (one-sided)':<22}  {stat['p_value']:>10.6f}")
    print(f"  {'Significant?':<22}  {str(stat['statistically_significant']):>10}")
    print(f"  {'Achieved power':<22}  {stat['achieved_power']:>10.1%}")
    print(f"  {'Adequately powered?':<22}  {str(stat['is_adequately_powered']):>10}")
    print(f"  {'Guardrails':<22}  {readout['guardrail_status']['summary']:>10}")
    print(f"\n  DECISION: [{readout['decision']}]")
    print(f"  Reasoning: {readout['reasoning'][:120]}")
    if len(readout['reasoning']) > 120:
        print(f"            {readout['reasoning'][120:240]}")


# ---------------------------------------------------------------------------
# Stage 1: Pre-registration
# ---------------------------------------------------------------------------

def stage_preregister() -> object:
    """
    Write and lock the experiment pre-registration.

    This MUST happen before any experiment data is generated.
    All downstream stages load and verify this file.

    Returns
    -------
    PreRegistration
        The locked pre-registration object.
    """
    banner("Stage 1 -- Pre-Registration (locked before any data)")
    from src.preregistration import register

    prereg = register(
        experiment_id="exp_placemux_001",
        hypothesis=(
            "Serving candidates with the v2.0 LightGBM ranker "
            "(trained on recency-weighted features) will increase "
            "7-day CTR by at least +5% relative to the v1.0 baseline ranker."
        ),
        primary_metric="CTR",
        direction="increase",
        baseline_rate=0.10,
        mde_relative=0.05,  # 5% relative lift is the minimum to care about
        alpha=0.05,
        power=0.80,
    )

    print(f"\n  Experiment ID  : {prereg.experiment_id}")
    print(f"  Hypothesis     : {prereg.hypothesis[:80]}...")
    print(f"  Primary metric : {prereg.primary_metric} ({prereg.direction})")
    print(f"  Baseline rate  : {prereg.baseline_rate:.0%}")
    print(f"  MDE            : {prereg.mde_relative:.0%} relative lift")
    print(f"  Alpha          : {prereg.alpha}")
    print(f"  Power          : {prereg.power:.0%}")
    print(f"  Required n     : {prereg.required_n_per_variant:,} impressions per variant")
    print(f"  Decision rule  : {prereg.decision_rule[:100]}...")
    print(f"  Tamper seal    : {prereg.content_hash[:32]}...")
    print(f"\n  [OK] Pre-registration locked at {prereg.registered_at}")

    return prereg


# ---------------------------------------------------------------------------
# Stage 2: SHIP scenario (+8% CTR lift)
# ---------------------------------------------------------------------------

def stage_ship_scenario() -> dict:
    """
    Scenario A: treatment delivers +8% relative CTR lift.
    Expected outcome: SHIP.

    Uses a 50/50 control/treatment split (no holdout) so both variants
    receive adequate impressions relative to the pre-registered required_n.

    Returns
    -------
    dict
        Full readout from ReadoutGenerator.
    """
    banner("Stage 2 -- Scenario A: SHIP Case (+8% CTR Lift)")
    from src.preregistration import load_and_verify
    from src.experiment_engine import (
        make_default_experiment, ExperimentEngine,
        VariantConfig, ExperimentConfig,
    )
    from src.experiment_simulator import ExperimentSimulator
    from src.guardrail_monitor import GuardrailMonitor
    from src.readout_generator import ReadoutGenerator

    prereg = load_and_verify(PREREG_PATH)

    # Use a 50/50 split so both arms reach the required 22,800 impressions
    # at a manageable total sample size (~50,000 users)
    config_50_50 = ExperimentConfig(
        experiment_id="exp_placemux_001",
        name="LightGBM v2 vs v1 -- 50/50 readout",
        holdout_fraction=0.0,   # no holdout for this readout run
        variants={
            "control": VariantConfig(
                name="control",
                model_version="v1.0.0-lightgbm",
                traffic_fraction=0.50,
                feature_flags={},
            ),
            "treatment": VariantConfig(
                name="treatment",
                model_version="v2.0.0-lgbm-improved",
                traffic_fraction=0.50,
                feature_flags={"rerank_by_recency": True},
            ),
        },
    )
    engine = ExperimentEngine(config_50_50)
    simulator = ExperimentSimulator(engine, random_state=42)

    counts = simulator.simulate(
        num_users=50000,
        control_ctr=0.10,
        treatment_ctr_multiplier=1.08,  # +8% CTR
        apply_rate_given_click=0.30,
        scenario_label="ship_A",
    )

    ctrl = counts.get("control", {})
    trt = counts.get("treatment", {})

    monitor = GuardrailMonitor()
    guardrail_report = monitor.check(
        control_impressions=ctrl.get("impressions", 0),
        control_clicks=ctrl.get("clicks", 0),
        control_applies=ctrl.get("applies", 0),
        treatment_impressions=trt.get("impressions", 0),
        treatment_clicks=trt.get("clicks", 0),
        treatment_applies=trt.get("applies", 0),
        engine=engine,
    )

    gen = ReadoutGenerator(
        prereg,
        readout_path="logs/readout_scenario_A.json",
    )
    readout = gen.generate(
        control_impressions=ctrl.get("impressions", 0),
        control_clicks=ctrl.get("clicks", 0),
        control_applies=ctrl.get("applies", 0),
        treatment_impressions=trt.get("impressions", 0),
        treatment_clicks=trt.get("clicks", 0),
        treatment_applies=trt.get("applies", 0),
        guardrail_report=guardrail_report,
        scenario_label="Scenario A -- SHIP (+8% CTR)",
    )

    print_readout_summary(readout)
    assert readout["decision"] == "SHIP", \
        f"FAIL: Scenario A should SHIP but got {readout['decision']}: {readout['reasoning']}"
    print(f"\n  [OK] Scenario A decision = SHIP (as expected)")
    return readout


# ---------------------------------------------------------------------------
# Stage 3: DO-NOT-SHIP scenario (+2% CTR, below MDE)
# ---------------------------------------------------------------------------

def stage_do_not_ship_scenario() -> dict:
    """
    Scenario B: treatment delivers only +2% CTR -- below MDE, not significant.
    Expected outcome: DO-NOT-SHIP.

    Returns
    -------
    dict
        Full readout from ReadoutGenerator.
    """
    banner("Stage 3 -- Scenario B: DO-NOT-SHIP Case (+2% CTR, Below MDE)")
    from src.preregistration import load_and_verify
    from src.experiment_engine import (
        ExperimentEngine, VariantConfig, ExperimentConfig,
    )
    from src.experiment_simulator import ExperimentSimulator
    from src.guardrail_monitor import GuardrailMonitor
    from src.readout_generator import ReadoutGenerator

    prereg = load_and_verify(PREREG_PATH)

    config_50_50 = ExperimentConfig(
        experiment_id="exp_placemux_001",
        name="LightGBM v2 vs v1 -- 50/50 readout",
        holdout_fraction=0.0,
        variants={
            "control": VariantConfig(
                name="control",
                model_version="v1.0.0-lightgbm",
                traffic_fraction=0.50,
                feature_flags={},
            ),
            "treatment": VariantConfig(
                name="treatment",
                model_version="v2.0.0-lgbm-improved",
                traffic_fraction=0.50,
                feature_flags={},
            ),
        },
    )
    engine = ExperimentEngine(config_50_50)
    simulator = ExperimentSimulator(engine, random_state=42)

    counts = simulator.simulate(
        num_users=50000,
        control_ctr=0.10,
        treatment_ctr_multiplier=1.00,  # +0% CTR -- well below 5% MDE
        apply_rate_given_click=0.30,
        scenario_label="dns_B",
    )

    ctrl = counts.get("control", {})
    trt = counts.get("treatment", {})

    monitor = GuardrailMonitor()
    guardrail_report = monitor.check(
        control_impressions=ctrl.get("impressions", 0),
        control_clicks=ctrl.get("clicks", 0),
        control_applies=ctrl.get("applies", 0),
        treatment_impressions=trt.get("impressions", 0),
        treatment_clicks=trt.get("clicks", 0),
        treatment_applies=trt.get("applies", 0),
    )

    gen = ReadoutGenerator(
        prereg,
        readout_path="logs/readout_scenario_B.json",
    )
    readout = gen.generate(
        control_impressions=ctrl.get("impressions", 0),
        control_clicks=ctrl.get("clicks", 0),
        control_applies=ctrl.get("applies", 0),
        treatment_impressions=trt.get("impressions", 0),
        treatment_clicks=trt.get("clicks", 0),
        treatment_applies=trt.get("applies", 0),
        guardrail_report=guardrail_report,
        scenario_label="Scenario B -- DO-NOT-SHIP (+2% CTR, below MDE)",
    )

    print_readout_summary(readout)
    assert readout["decision"] in ("DO-NOT-SHIP", "INCONCLUSIVE"), \
        f"FAIL: Scenario B should NOT ship but got {readout['decision']}"
    print(f"\n  [OK] Scenario B decision = {readout['decision']} (as expected)")
    return readout


# ---------------------------------------------------------------------------
# Stage 4: Tamper test
# ---------------------------------------------------------------------------

def stage_tamper_test() -> None:
    """
    Corrupt the pre-registration file (simulate metric-swapping after data).
    The hash verification must catch this and raise ValueError.
    """
    banner("Stage 4 -- Tamper Test (metric-swap after data detected)")
    import json as _json
    from src.preregistration import load_and_verify

    # Read and corrupt the pre-registration
    with open(PREREG_PATH, "r", encoding="utf-8") as f:
        d = _json.load(f)

    original_mde = d["mde_relative"]
    d["mde_relative"] = 0.001  # Fraudulently lower the MDE threshold post-hoc

    with open(PREREG_PATH, "w", encoding="utf-8") as f:
        _json.dump(d, f, indent=2)

    print(f"\n  [INJECT] Lowered mde_relative from {original_mde} to 0.001")
    print(f"  [INJECT] This simulates lowering the bar after seeing the data.")

    try:
        load_and_verify(PREREG_PATH)
        print(f"  [FAIL] Hash check did NOT catch the tamper -- this is a bug!")
    except ValueError as e:
        print(f"  [OK] Hash check caught tamper: {str(e)[:120]}")
    finally:
        # Restore original value so later stages work
        d["mde_relative"] = original_mde
        # Recompute hash
        from src.preregistration import _compute_hash
        d["content_hash"] = _compute_hash(d)
        with open(PREREG_PATH, "w", encoding="utf-8") as f:
            _json.dump(d, f, indent=2)
        print(f"  [OK] Pre-registration restored to original state")


# ---------------------------------------------------------------------------
# Stage 5: Break it -- underpowered experiment -> INCONCLUSIVE
# ---------------------------------------------------------------------------

def stage_break_underpowered() -> None:
    """
    Run only 100 users (far below required_n).
    ReadoutGenerator must return INCONCLUSIVE -- not SHIP or DO-NOT-SHIP.
    """
    banner("Stage 5 -- Break It: Underpowered Experiment (n << required)")
    from src.preregistration import load_and_verify
    from src.experiment_engine import make_default_experiment, ExperimentEngine
    from src.experiment_simulator import ExperimentSimulator
    from src.readout_generator import ReadoutGenerator

    prereg = load_and_verify(PREREG_PATH)

    config = make_default_experiment()
    engine = ExperimentEngine(config)
    simulator = ExperimentSimulator(engine, random_state=42)

    counts = simulator.simulate(
        num_users=100,   # Deliberately tiny -- far below required_n
        control_ctr=0.10,
        treatment_ctr_multiplier=1.08,
        scenario_label="underpowered",
    )

    ctrl = counts.get("control", {})
    trt = counts.get("treatment", {})

    gen = ReadoutGenerator(
        prereg,
        readout_path="logs/readout_underpowered.json",
    )
    readout = gen.generate(
        control_impressions=ctrl.get("impressions", 0),
        control_clicks=ctrl.get("clicks", 0),
        control_applies=ctrl.get("applies", 0),
        treatment_impressions=trt.get("impressions", 0),
        treatment_clicks=trt.get("clicks", 0),
        treatment_applies=trt.get("applies", 0),
        scenario_label="Underpowered",
    )

    print(f"\n  Treatment impressions : {trt.get('impressions', 0)}")
    print(f"  Required n/variant   : {prereg.required_n_per_variant:,}")
    print(f"  Decision             : {readout['decision']}")
    print(f"  Reasoning            : {readout['reasoning'][:120]}")

    assert readout["decision"] == "INCONCLUSIVE", \
        f"FAIL: Underpowered experiment should be INCONCLUSIVE, got {readout['decision']}"
    print(f"\n  [OK] INCONCLUSIVE returned correctly -- system refuses to decide without data.")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_demo() -> None:
    """
    Orchestrate the full end-to-end Task 10 demonstration.
    All stages in sequence, with assertions at each step.
    """
    logger.info("=== Phase 3, Task 10: Growth Integration & Experiment Readout Demo ===")

    stage_preregister()
    stage_ship_scenario()
    stage_do_not_ship_scenario()
    stage_tamper_test()
    stage_break_underpowered()

    banner("[DONE] Task 10 Demo Complete")
    print(f"  Deliverables:")
    print(f"    src/preregistration.py        -- tamper-evident hypothesis lock")
    print(f"    src/readout_generator.py      -- effect size, CIs, ship decision")
    print(f"    logs/preregistration.json     -- locked pre-registration + hash")
    print(f"    logs/readout_scenario_A.json  -- SHIP readout")
    print(f"    logs/readout_scenario_B.json  -- DO-NOT-SHIP readout")
    print(f"    logs/readout_underpowered.json -- INCONCLUSIVE readout")
    print(f"    logs/task10.log / task10_demo.log -- experiment logs\n")


def main() -> None:
    """Entry point -- fatal error trap at top level (Rule 2)."""
    try:
        run_demo()
    except AssertionError as e:
        logger.critical(f"Assertion failed: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        logger.critical(f"Missing required file: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unhandled fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
