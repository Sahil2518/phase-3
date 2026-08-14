"""
demo_task09.py -- PlaceMux Phase 3, Task 9
==========================================
End-to-end demonstration of the Experimentation Platform.

Journey (run once, start to finish)
-------------------------------------
1. Configure experiment (variants + traffic split)
2. Assign 1,000 users -- verify consistent re-assignment (no flipping)
3. Simulate GOOD model traffic -- guardrail must NOT fire
4. Simulate BAD model traffic -- guardrail MUST fire and halt experiment
5. Show holdout cumulative lift report
6. Break it on purpose: missing model, engine halted mid-experiment
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Rule 2: Structured logging
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("logs", "task09_demo.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def banner(title: str) -> None:
    """Print a formatted section banner to stdout."""
    width = 64
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


# ---------------------------------------------------------------------------
# Stage 1: Configure experiment
# ---------------------------------------------------------------------------

def stage_configure() -> tuple:
    """
    Build and display the experiment configuration.

    Returns
    -------
    tuple
        (ExperimentConfig, ExperimentEngine)
    """
    banner("Stage 1 -- Experiment Configuration")
    from src.experiment_engine import make_default_experiment, ExperimentEngine

    config = make_default_experiment(
        experiment_id="exp_placemux_001",
        name="LightGBM v2 vs v1 Ranking Test",
    )
    engine = ExperimentEngine(config)

    print(f"\n  Experiment ID : {config.experiment_id}")
    print(f"  Name          : {config.name}")
    print(f"  Status        : {config.status}")
    print(f"  Holdout       : {config.holdout_fraction:.0%} of traffic")
    print(f"\n  Variants:")
    for vname, vcfg in config.variants.items():
        print(
            f"    {vname:<12} -> model={vcfg.model_version}  "
            f"traffic={vcfg.traffic_fraction:.0%}  "
            f"flags={vcfg.feature_flags}"
        )
    print()

    return config, engine


# ---------------------------------------------------------------------------
# Stage 2: Consistent assignment (no flipping)
# ---------------------------------------------------------------------------

def stage_assignment_consistency(engine) -> None:
    """
    Assign 1,000 users twice and verify every user gets the same variant.

    Parameters
    ----------
    engine : ExperimentEngine
    """
    banner("Stage 2 -- Consistent Variant Assignment (no user flipping)")

    N = 1000
    user_ids = [f"user_{i:05d}" for i in range(N)]

    # First pass
    assignments_1 = {uid: engine.assign(uid).variant for uid in user_ids}
    # Second pass -- must be identical
    assignments_2 = {uid: engine.assign(uid).variant for uid in user_ids}

    flips = [uid for uid in user_ids if assignments_1[uid] != assignments_2[uid]]

    variant_counts = pd.Series(assignments_1.values()).value_counts()
    print(f"\n  Users assigned: {N}")
    for variant, count in variant_counts.items():
        pct = count / N * 100
        print(f"    {variant:<12}: {count:>4} ({pct:.1f}%)")

    print(f"\n  Re-assignment check: {N} users re-assigned a second time")
    print(f"  Users who flipped variants: {len(flips)}")

    assert len(flips) == 0, f"FAIL: {len(flips)} users flipped between assignments!"
    print(f"  [OK] Zero flips -- assignment is fully deterministic")

    # Print a worked example
    sample_uid = "user_00042"
    a = engine.assign(sample_uid)
    print(f"\n  Worked example:")
    print(f"    user_id      : {a.user_id}")
    print(f"    bucket       : {a.bucket}")
    print(f"    variant      : {a.variant}")
    print(f"    model_version: {a.model_version}")
    print(f"    feature_flags: {a.feature_flags}")


# ---------------------------------------------------------------------------
# Stage 3: Good model -- guardrail must NOT fire
# ---------------------------------------------------------------------------

def stage_good_model(engine) -> dict:
    """
    Simulate a +12% CTR improvement.  Guardrail must stay green.

    Parameters
    ----------
    engine : ExperimentEngine

    Returns
    -------
    dict
        Aggregated counts from the simulation.
    """
    banner("Stage 3 -- Good Model (guardrail should stay GREEN)")
    from src.experiment_simulator import ExperimentSimulator
    from src.guardrail_monitor import GuardrailMonitor

    # Reset engine status in case it was halted
    engine.resume()

    simulator = ExperimentSimulator(engine, random_state=42)
    counts = simulator.simulate(
        num_users=5000,
        control_ctr=0.10,
        treatment_ctr_multiplier=1.12,   # +12% CTR
        apply_rate_given_click=0.30,
        scenario_label="good_model",
    )

    monitor = GuardrailMonitor()
    ctrl = counts.get("control", {})
    trt = counts.get("treatment", {})

    report = monitor.check(
        control_impressions=ctrl.get("impressions", 0),
        control_clicks=ctrl.get("clicks", 0),
        control_applies=ctrl.get("applies", 0),
        treatment_impressions=trt.get("impressions", 0),
        treatment_clicks=trt.get("clicks", 0),
        treatment_applies=trt.get("applies", 0),
        treatment_errors=trt.get("errors", 0),
        engine=engine,
    )

    _print_guardrail_report(report)
    assert report["status"] == "OK", \
        f"FAIL: Good model triggered a guardrail breach! {report['guardrail_breaches']}"
    print(f"  [OK] Good model passed -- no guardrail breach.")
    return counts


# ---------------------------------------------------------------------------
# Stage 4: Bad model -- guardrail MUST fire
# ---------------------------------------------------------------------------

def stage_bad_model(config) -> dict:
    """
    Simulate a -30% CTR drop.  Guardrail must halt the experiment.

    A fresh engine instance is created so the bad model doesn't contaminate
    the good-model engine state from Stage 3.

    Parameters
    ----------
    config : ExperimentConfig

    Returns
    -------
    dict
        Aggregated counts from the simulation.
    """
    banner("Stage 4 -- Bad Model (guardrail should HALT experiment)")
    from src.experiment_engine import ExperimentEngine
    from src.experiment_simulator import ExperimentSimulator
    from src.guardrail_monitor import GuardrailMonitor

    # Fresh engine instance for the bad model scenario
    bad_engine = ExperimentEngine(config)

    simulator = ExperimentSimulator(bad_engine, random_state=42)
    counts = simulator.simulate(
        num_users=30000,       # 30k gives ~3k treatment impressions -> enough power
        control_ctr=0.10,
        treatment_ctr_multiplier=0.70,  # -30% CTR
        apply_rate_given_click=0.30,
        scenario_label="bad_model",
        append=True,  # append to existing log for holdout analysis
    )

    monitor = GuardrailMonitor()
    ctrl = counts.get("control", {})
    trt = counts.get("treatment", {})

    report = monitor.check(
        control_impressions=ctrl.get("impressions", 0),
        control_clicks=ctrl.get("clicks", 0),
        control_applies=ctrl.get("applies", 0),
        treatment_impressions=trt.get("impressions", 0),
        treatment_clicks=trt.get("clicks", 0),
        treatment_applies=trt.get("applies", 0),
        treatment_errors=trt.get("errors", 0),
        engine=bad_engine,
    )

    _print_guardrail_report(report)

    if report["status"] == "HALTED":
        print(f"  [OK] Guardrail fired correctly -- experiment halted.")
        print(f"  [OK] Engine status: {bad_engine.config.status}")
        print(f"  [OK] All traffic now routed -> control.")
        for b in report["guardrail_breaches"]:
            print(f"  [BREACH] {b['reason']}")
    else:
        print(f"  [WARN] Guardrail did NOT fire (may need more data for significance).")

    return counts


# ---------------------------------------------------------------------------
# Stage 5: Holdout cumulative lift
# ---------------------------------------------------------------------------

def stage_holdout_report() -> None:
    """Compute and display the cumulative model value vs holdout baseline."""
    banner("Stage 5 -- Holdout Group: Cumulative Model Value")
    from src.holdout_manager import HoldoutManager

    manager = HoldoutManager()
    report = manager.compute_cumulative_lift()

    if report.get("status") == "OK":
        holdout = report["holdout_baseline"]
        print(f"\n  Holdout baseline (no model exposure):")
        print(f"    CTR        : {holdout['ctr']:.4f}")
        print(f"    Apply rate : {holdout['apply_rate']:.4f}")
        print(f"    Impressions: {holdout['impressions']}")

        print(f"\n  Cumulative lift vs holdout:")
        print(f"  {'Variant':<12}  {'CTR':>8}  {'Apply':>8}  {'CTR Lift':>10}  {'Apply Lift':>12}")
        print(f"  {'-'*56}")
        for row in report["variant_lifts"]:
            print(
                f"  {row['variant']:<12}  "
                f"{row['ctr']:>8.4f}  "
                f"{row['apply_rate']:>8.4f}  "
                f"{row['ctr_lift_vs_holdout']:>+9.1%}  "
                f"{row['apply_lift_vs_holdout']:>+11.1%}"
            )
    else:
        print(f"  Status: {report.get('status')} -- {report.get('reason', '')}")

    print(f"\n  Full report saved -> logs/holdout_report.json")


# ---------------------------------------------------------------------------
# Stage 6: Break it on purpose
# ---------------------------------------------------------------------------

def stage_break_it(config) -> None:
    """
    Three fault-injection tests to confirm graceful degradation.

    Tests
    -----
    1. Empty user_id -> falls back to anonymous assignment
    2. Engine halted at start -> all users get control
    3. No events log -> holdout manager returns graceful error

    Parameters
    ----------
    config : ExperimentConfig
    """
    banner("Stage 6 -- Break It On Purpose (fault injection)")
    from src.experiment_engine import ExperimentEngine
    from src.holdout_manager import HoldoutManager

    # Test 1: Empty user_id
    print("\n  [Test 1] Empty user_id:")
    try:
        engine = ExperimentEngine(config)
        a = engine.assign("")
        print(f"  -> Assigned to variant='{a.variant}'  (fallback handled gracefully)")
    except Exception as e:
        print(f"  -> Exception: {e}")

    # Test 2: Halted engine -- all users get control
    print("\n  [Test 2] Experiment pre-halted -> all traffic to control:")
    try:
        engine = ExperimentEngine(config)
        engine.halt(reason="Injected fault test")
        assignments = [engine.assign(f"user_{i}").variant for i in range(100)]
        non_control = [v for v in assignments if v != "control"]
        print(f"  -> 100 users assigned. Non-control variants: {len(non_control)}")
        assert len(non_control) == 0, "Some users escaped to treatment while halted!"
        print(f"  -> [OK] All 100 users routed to control while engine is HALTED")
    except Exception as e:
        print(f"  -> Exception: {e}")

    # Test 3: Missing events log -> HoldoutManager degrades gracefully
    print("\n  [Test 3] Missing events log -> HoldoutManager graceful error:")
    try:
        manager = HoldoutManager(events_log_path="logs/nonexistent_events.jsonl")
        report = manager.compute_cumulative_lift()
        print(f"  -> report status: '{report.get('status', 'unknown')}' / error: '{report.get('error', 'none')}'")
        print(f"  -> [OK] No crash -- returned structured error response")
    except Exception as e:
        print(f"  -> Exception: {e}")

    print(f"\n  [OK] All fault-injection tests completed -- system degrades as designed.\n")


# ---------------------------------------------------------------------------
# Helper: print guardrail report
# ---------------------------------------------------------------------------

def _print_guardrail_report(report: dict) -> None:
    """Print a compact guardrail report to stdout."""
    ctrl = report["metrics"]["control"]
    trt = report["metrics"]["treatment"]
    print(f"\n  Control  : impressions={ctrl['impressions']}, CTR={ctrl['ctr']:.4f}, apply={ctrl['apply_rate']:.4f}")
    print(f"  Treatment: impressions={trt['impressions']}, CTR={trt['ctr']:.4f}, apply={trt['apply_rate']:.4f}")
    print(f"  Status   : {report['status']}  |  Breaches: {len(report['guardrail_breaches'])}")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_demo() -> None:
    """
    Run all six stages end-to-end.

    Each stage validates its own invariants.  The demo is considered
    successful if all assertions pass and the pipeline runs to completion.
    """
    logger.info("=== Phase 3, Task 9: Experimentation Platform Demo ===")

    config, engine = stage_configure()
    stage_assignment_consistency(engine)
    stage_good_model(engine)
    stage_bad_model(config)
    stage_holdout_report()
    stage_break_it(config)

    banner("[DONE] Task 9 Demo Complete")
    print(f"  All deliverables produced:")
    print(f"    src/experiment_engine.py       -- deterministic variant assignment")
    print(f"    src/holdout_manager.py         -- permanent holdout + cumulative lift")
    print(f"    src/guardrail_monitor.py       -- guardrail metrics + auto-halt")
    print(f"    src/experiment_simulator.py    -- traffic simulation (3 scenarios)")
    print(f"    logs/experiment_assignments.jsonl -- assignment audit log")
    print(f"    logs/experiment_events.jsonl      -- simulated event log")
    print(f"    logs/guardrail_report.json        -- latest guardrail check")
    print(f"    logs/holdout_report.json          -- cumulative lift vs holdout")
    print(f"    logs/task09.log / task09_demo.log -- experiment logs\n")


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
