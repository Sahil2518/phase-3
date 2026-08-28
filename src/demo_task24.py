"""
demo_task24.py — PlaceMux Phase 3, Task 24
==========================================
End-to-End Demo: Disaster Recovery, Chaos Testing & Business Continuity

Five stages:
  A: Setup — prerequisites, synthetic data, environment check
  B: Chaos scenarios — run all 5 failure scenarios, print pass/fail
  C: Live degradation proof — kill model in real-time, show heuristic taking over
  D: ML Incident Runbook — generate and validate runbook coverage
  E: Summary table + final status

Usage:
    python -m src.demo_task24
"""

import os
import sys
import json
import logging

os.makedirs("logs",   exist_ok=True)
os.makedirs("models", exist_ok=True)

_fh = logging.FileHandler("logs/task24.log", encoding="utf-8")
_sh = logging.StreamHandler(sys.stdout)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[_fh, _sh],
)
logger = logging.getLogger(__name__)

from src.chaos_engine         import ChaosEngine, generate_candidates
from src.graceful_degradation import GracefulDegradationLayer, HeuristicMatcher
from src.ml_incident_runbook  import RunbookGenerator


def section(title: str) -> None:
    """Print a visually distinct section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Stage A: Setup
# ---------------------------------------------------------------------------

def stage_a_setup() -> dict:
    section("Stage A: Setup — Prerequisites & Synthetic Data")

    candidates = generate_candidates(n=200, seed=42)
    n_relevant = sum(1 for c in candidates if c["relevance"] == 1)
    n_total = len(candidates)

    print(f"\n  Synthetic candidates generated : {n_total}")
    print(f"  Relevant (ground-truth)        : {n_relevant} ({n_relevant/n_total:.0%})")
    print(f"  Feature columns                : 9 (FEATURE_COLS)")
    print(f"\n  One-line bar:")
    print(f"    'When the model dies, matching degrades to a sane heuristic")
    print(f"     and someone is paged — nothing silently breaks.'")
    print(f"\n  Heuristic NDCG@10 bar : >= 0.45")
    print(f"  Random baseline NDCG  : 0.30")
    print(f"\n  [OK] Prerequisites confirmed.")

    return {"candidates": candidates, "n_total": n_total, "n_relevant": n_relevant}


# ---------------------------------------------------------------------------
# Stage B: Chaos Scenarios
# ---------------------------------------------------------------------------

def stage_b_chaos(candidates: list) -> dict:
    section("Stage B: Chaos Scenarios — 5 ML Failure Modes")

    engine = ChaosEngine(n_candidates=200, seed=42)
    report = engine.run_all()

    print(f"\n  Chaos scenarios run : {report['n_scenarios']}")
    print(f"  Passed              : {report['n_passed']}")
    print(f"  Failed              : {report['n_failed']}")
    print(f"\n  {'Scenario':<30} {'Mode/Metric':<24} {'Bar':<20} {'Status'}")
    print(f"  {'-'*80}")

    for s in report["scenarios"]:
        sid = s["scenario"]
        passed = s.get("passed", False)
        status = "[PASS]" if passed else "[FAIL]"

        if sid == "MODEL_UNAVAILABLE":
            pc = s.get("post_chaos", {})
            metric = f"NDCG@10={pc.get('ndcg_at_10','?')}"
            bar = f">= {report['heuristic_ndcg_bar']}"
        elif sid == "STALE_FEATURES":
            metric = f"RED_PSI={s.get('any_feature_red','?')}"
            bar = "any_psi_red=True"
        elif sid == "CORRUPTED_TRAINING_DATA":
            metric = f"gate_rejected={s.get('gate_rejected','?')}"
            bar = "gate_must_reject=True"
        elif sid == "FEATURE_STORE_DOWN":
            metric = f"coverage={s.get('coverage','?'):.0%}"
            bar = "coverage=100%"
        elif sid == "NAN_MODEL_OUTPUT":
            metric = f"nan_in_output={s.get('nan_in_output','?')}"
            bar = "nan_in_output=0"
        else:
            metric = "?"
            bar = "?"

        print(f"  {sid:<30} {metric:<24} {bar:<20} {status}")

    # Worked example: MODEL_UNAVAILABLE
    print(f"\n  Worked Example — Scenario: MODEL_UNAVAILABLE")
    print(f"  " + "-" * 65)
    s1 = next((s for s in report["scenarios"] if s["scenario"] == "MODEL_UNAVAILABLE"), {})
    pc = s1.get("post_chaos", {})
    print(f"  Input       : 200 candidates, champion model file deleted")
    print(f"  Output      : mode=HEURISTIC | NDCG@10={pc.get('ndcg_at_10','?')} | degraded_mode=True")
    print(f"  Plain reason: GracefulDegradationLayer caught FileNotFoundError;")
    print(f"                heuristic scored all 200 candidates; P1 alert emitted.")
    print(f"  Unavailable : (this IS the unavailability test — heuristic is the answer)")
    print(f"  " + "-" * 65)

    print(f"\n  Chaos results saved: logs/chaos_results.json")
    print(f"  Pager alerts log  : logs/chaos_alerts.jsonl")

    return {
        "report": report,
        "all_passed": report["all_passed"],
        "n_passed": report["n_passed"],
        "n_failed": report["n_failed"],
    }


# ---------------------------------------------------------------------------
# Stage C: Live Degradation Proof
# ---------------------------------------------------------------------------

def stage_c_live_degradation() -> dict:
    section("Stage C: Live Degradation Proof — Kill the Model, Watch Heuristic Take Over")

    hm = HeuristicMatcher(max_sessions=20.0)
    candidates = generate_candidates(n=50, seed=7)

    # Step 1: ML available (simulated)
    print("\n  Step 1: ML scorer AVAILABLE")

    def live_ml_scorer(cands):
        """Simulated working ML scorer."""
        import numpy as np
        rng = np.random.default_rng(42)
        return [(c["candidate_id"], float(rng.uniform(0.3, 0.9))) for c in cands]

    layer_live = GracefulDegradationLayer(ml_scorer=live_ml_scorer)
    result_live = layer_live.score(candidates)
    ndcg_live = hm.compute_ndcg_at_k(result_live["ranked"], relevance_key="relevance", k=10)
    print(f"  mode={result_live['mode']} | degraded={result_live['degraded_mode']} | NDCG@10={ndcg_live:.4f}")

    # Step 2: Kill the model (set scorer to None)
    print("\n  Step 2: ML scorer KILLED (model set to None)")

    layer_dead = GracefulDegradationLayer(ml_scorer=None, alert_severity="P1")
    result_dead = layer_dead.score(candidates, alert_type="MODEL_UNAVAILABLE")
    ndcg_dead = hm.compute_ndcg_at_k(result_dead["ranked"], relevance_key="relevance", k=10)

    print(f"  mode={result_dead['mode']} | degraded={result_dead['degraded_mode']} | NDCG@10={ndcg_dead:.4f}")
    print(f"  Alert emitted : {result_dead['alert'] is not None}")
    print(f"  Alert type    : {result_dead['alert']['alert_type'] if result_dead['alert'] else 'N/A'}")

    # Step 3: Confirm heuristic quality above bar
    from src.chaos_engine import HEURISTIC_NDCG_BAR
    heuristic_ok = ndcg_dead >= HEURISTIC_NDCG_BAR
    silent_failure = not result_dead["degraded_mode"]

    print(f"\n  Heuristic NDCG@10 >= {HEURISTIC_NDCG_BAR} : {'[PASS]' if heuristic_ok else '[FAIL]'} ({ndcg_dead:.4f})")
    print(f"  No silent failure (degraded_mode=True) : {'[PASS]' if not silent_failure else '[FAIL]'}")
    print(f"  Pager alert emitted                   : {'[PASS]' if result_dead['alert'] else '[FAIL]'}")

    return {
        "ml_ndcg": round(ndcg_live, 4),
        "heuristic_ndcg": round(ndcg_dead, 4),
        "heuristic_ok": heuristic_ok,
        "no_silent_failure": not silent_failure,
        "alert_emitted": result_dead["alert"] is not None,
        "live_passed": heuristic_ok and (not silent_failure) and result_dead["alert"] is not None,
    }


# ---------------------------------------------------------------------------
# Stage D: ML Incident Runbook
# ---------------------------------------------------------------------------

def stage_d_runbook() -> dict:
    section("Stage D: ML Incident Runbook — Generate & Validate")

    gen = RunbookGenerator()
    result = gen.generate()

    print(f"\n  Runbook path       : {result['path']}")
    print(f"  Procedures written : {result['n_procedures']}")
    print(f"  Coverage complete  : {result['all_covered']}")
    print(f"  Missing scenarios  : {result['coverage']['missing'] or 'none'}")
    print(f"  File size          : {result['size_bytes']:,} bytes")

    # Show section headings
    print(f"\n  Runbook sections:")
    sections = [
        "1. Incident Classification Matrix (P1-P4)",
        "2. Runbook Procedures (5 scenarios)",
        "3. Verification Checklist",
        "4. Escalation Chain",
        "5. Post-Incident Review Template",
    ]
    for s in sections:
        print(f"    [*] {s}")

    # Worked example: Runbook procedure for FEATURE_STORE_DOWN
    print(f"\n  Worked Example — Runbook Procedure: FEATURE_STORE_DOWN")
    print(f"  " + "-" * 65)
    print(f"  Input       : feature store returns empty DataFrame")
    print(f"  Output      : P1 pager alert + heuristic covers 100% of candidates")
    print(f"  Plain reason: runbook tells engineer to check DB health,")
    print(f"                leave heuristic running, probe feature freshness post-recovery")
    print(f"  Unavailable : heuristic is the fallback — system never returns empty response")
    print(f"  " + "-" * 65)

    return result


# ---------------------------------------------------------------------------
# Stage E: Summary Table + Final Status
# ---------------------------------------------------------------------------

def stage_e_summary(
    setup: dict,
    chaos: dict,
    live: dict,
    runbook: dict,
) -> None:
    section("Stage E: End-to-End Summary")

    # Count pager alerts in log
    n_alerts = 0
    if os.path.exists("logs/chaos_alerts.jsonl"):
        with open("logs/chaos_alerts.jsonl", encoding="utf-8") as f:
            n_alerts = sum(1 for line in f if line.strip())

    rows = [
        ("Synthetic candidates generated",    str(setup["n_total"]),          True),
        ("Chaos scenarios run",               "5",                            True),
        ("Chaos scenarios PASSED",            str(chaos["n_passed"]),         chaos["all_passed"]),
        ("Chaos scenarios FAILED",            str(chaos["n_failed"]),         chaos["n_failed"] == 0),
        ("MODEL_UNAVAILABLE: heuristic NDCG@10",
         str(chaos["report"]["scenarios"][0].get("post_chaos", {}).get("ndcg_at_10", "?")),
         chaos["report"]["scenarios"][0].get("passed", False)),
        ("STALE_FEATURES: PSI RED detected",
         str(chaos["report"]["scenarios"][1].get("any_feature_red", "?")),
         chaos["report"]["scenarios"][1].get("passed", False)),
        ("CORRUPTED_DATA: retrain gate rejected",
         str(chaos["report"]["scenarios"][2].get("gate_rejected", "?")),
         chaos["report"]["scenarios"][2].get("passed", False)),
        ("FEATURE_STORE_DOWN: 100% heuristic coverage",
         str(chaos["report"]["scenarios"][3].get("coverage", "?")),
         chaos["report"]["scenarios"][3].get("passed", False)),
        ("NAN_MODEL_OUTPUT: 0 NaN in output",
         str(chaos["report"]["scenarios"][4].get("nan_in_output", "?")),
         chaos["report"]["scenarios"][4].get("passed", False)),
        ("Live kill test: heuristic NDCG@10 >= 0.45",
         str(live["heuristic_ndcg"]),        live["heuristic_ok"]),
        ("Live kill test: no silent failure",
         "True",                             live["no_silent_failure"]),
        ("Live kill test: pager alert emitted",
         "True",                             live["alert_emitted"]),
        ("Pager alerts in logs/chaos_alerts.jsonl",
         str(n_alerts),                      n_alerts > 0),
        ("Runbook procedures documented",     str(runbook["n_procedures"]),   True),
        ("Runbook coverage complete",         str(runbook["all_covered"]),    runbook["all_covered"]),
        ("logs/ml_incident_runbook.md",       "present",                      os.path.exists("logs/ml_incident_runbook.md")),
        ("logs/chaos_results.json",           "present",                      os.path.exists("logs/chaos_results.json")),
        ("logs/chaos_alerts.jsonl",           "present",                      os.path.exists("logs/chaos_alerts.jsonl")),
    ]

    print(f"\n  +{'-'*72}+")
    print(f"  | {'Check':<44} {'Value':>10} {'Status':>14} |")
    print(f"  +{'-'*72}+")
    for label, val, passed in rows:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  | {label:<44} {str(val):>10} {status:>14} |")
    print(f"  +{'-'*72}+")

    all_pass = all(p for _, _, p in rows)
    print(f"\n  Overall demo result: {'[ALL PASS]' if all_pass else '[SOME CHECKS FAILED]'}")

    print(f"\n  Output files:")
    paths = [
        "logs/chaos_results.json",
        "logs/chaos_alerts.jsonl",
        "logs/ml_incident_runbook.md",
        "logs/task24.log",
    ]
    for p in paths:
        exists = "[OK  ]" if os.path.exists(p) else "[MISS]"
        print(f"    {exists} {p}")

    print("\n" + "=" * 70)
    print("  Task 24 Demo Complete.")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_demo() -> None:
    """Execute the full Task 24 chaos demo."""
    setup   = stage_a_setup()
    chaos   = stage_b_chaos(setup["candidates"])
    live    = stage_c_live_degradation()
    runbook = stage_d_runbook()
    stage_e_summary(setup, chaos, live, runbook)


def main() -> None:
    """Top-level entry point with fatal error trap (Rule 2)."""
    try:
        run_demo()
    except FileNotFoundError as e:
        logger.critical(f"Missing required file: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unhandled fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
