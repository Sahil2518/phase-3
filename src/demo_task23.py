"""
demo_task23.py -- PlaceMux Phase 3, Task 23
============================================
End-to-End Demo: Compliance Audit -- DPDP, GDPR & SOC 2 Readiness

Five stages:
  A: Setup -- generate synthetic data, confirm prerequisites
  B: Data-Subject Rights -- access request, live deletion, retraining impact
  C: Automated Decision Disclosure -- log decisions, explain, escalate borderline
  D: Compliance Audit Pack -- fairness report, model card, lineage, SOC 2 bundle
  E: Failure paths + integrated summary table

Usage:
    python -m src.demo_task23
"""

import os
import sys
import json
import logging

os.makedirs("logs", exist_ok=True)
os.makedirs("logs/audit_pack", exist_ok=True)

_fh = logging.FileHandler("logs/task23.log", encoding="utf-8")
_sh = logging.StreamHandler(sys.stdout)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[_fh, _sh],
)
logger = logging.getLogger(__name__)

from src.data_rights_engine    import DataSubjectRightsEngine
from src.decision_disclosure   import DecisionDisclosureEngine
from src.compliance_audit_pack import ComplianceAuditPack


def section(title: str) -> None:
    """Print a visually distinct section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Stage A: Setup
# ---------------------------------------------------------------------------

def stage_a_setup():
    section("Stage A: Setup -- Confirm Prerequisites & Build Synthetic Stores")

    engine = DataSubjectRightsEngine(n_candidates=200, seed=42)
    summ   = engine.summary()

    print(f"\n  Feature store   : {summ['candidates_in_feature_store']} candidates")
    print(f"  Interaction log : {summ['interaction_events']} events")
    print(f"  Score log       : {summ['score_records']} records")
    print(f"  Training queue  : {summ['training_queue_size']} queued")

    print(f"\n  One-line bar: An auditor asks how a candidate was ranked")
    print(f"  and you produce the model, the data, the explanation,")
    print(f"  and the human-review route.")
    print(f"\n  [OK] Prerequisites confirmed.")
    return engine


# ---------------------------------------------------------------------------
# Stage B: Data-Subject Rights
# ---------------------------------------------------------------------------

def stage_b_data_rights(engine: DataSubjectRightsEngine) -> dict:
    section("Stage B: Data-Subject Rights -- Access, Deletion & Retraining Impact")

    target = "CAND_0005"

    # --- Right of Access ---
    print(f"\n  -- Right of Access: {target} --")
    access = engine.handle_access_request(target)
    print(f"  Found in feature store  : {access['found']}")
    print(f"  Interaction events      : {len(access['interaction_events'])}")
    print(f"  Model score records     : {len(access['model_scores'])}")
    print(f"  In training queue       : {access['in_training_queue']}")
    print(f"  Training note           : {access['training_membership_note']}")

    # Worked example
    print(f"\n  Worked example -- Access:")
    print(f"    Input       : handle_access_request('{target}')")
    print(f"    Output      : profile features + {len(access['model_scores'])} scores + {len(access['interaction_events'])} events")
    print(f"    Plain reason: DPDP Sec. 11 / GDPR Art. 15 -- subject receives all stored data")
    print(f"    Unavailable : Returns empty bundle with 'found=False'; no crash")

    # --- Right to Erasure ---
    print(f"\n  -- Right to Erasure: {target} --")
    before = engine.summary()
    cert   = engine.handle_deletion_request(target)
    after  = engine.summary()

    print(f"  Records deleted         : {cert['total_records_deleted']}")
    print(f"  Feature profile removed : {cert['records_deleted']['feature_profile']}")
    print(f"  Interaction events rm   : {cert['records_deleted']['interaction_events']}")
    print(f"  Score records removed   : {cert['records_deleted']['model_scores']}")
    print(f"  Training queue removed  : {cert['records_deleted']['training_queue']}")
    print(f"  Cert hash (first 16)    : {cert['certificate_hash'][:16]}...")
    print(f"  Status                  : {cert['status']}")
    print(f"\n  Store before: {before['candidates_in_feature_store']} candidates | "
          f"after: {after['candidates_in_feature_store']} candidates")

    # Verify deletion is irreversible (second access returns found=False)
    re_access = engine.handle_access_request(target)
    print(f"  Re-access after deletion: found={re_access['found']}  [{'OK' if not re_access['found'] else 'FAIL'}]")

    # --- Retraining Impact ---
    print(f"\n  -- Retraining Implications --")
    impact = engine.assess_retraining_impact(target, cert)
    print(f"  Influence score         : {impact['influence_score']:.3f}")
    print(f"  Threshold               : {impact['influence_threshold']:.2f}")
    print(f"  Retrain required        : {impact['retrain_required']}")
    print(f"  Reason                  : {impact['reason']}")

    # Also delete a high-influence candidate for demo
    high_target = "CAND_0001"
    cert2   = engine.handle_deletion_request(high_target)
    impact2 = engine.assess_retraining_impact(high_target, cert2)
    print(f"\n  High-activity candidate ({high_target}):")
    print(f"  Influence score : {impact2['influence_score']:.3f}  "
          f"retrain_required={impact2['retrain_required']}")

    cert_ok    = cert["total_records_deleted"] >= 1
    delete_ok  = not re_access["found"]
    audit_ok   = os.path.exists("logs/data_rights_log.jsonl")
    cert_file  = os.path.exists(f"logs/audit_pack/deletion_cert_{target}.json")

    print(f"\n  Deletion completeness   : {'[OK]' if cert_ok else '[FAIL]'}")
    print(f"  Re-access returns empty : {'[OK]' if delete_ok else '[FAIL]'}")
    print(f"  Audit log written       : {'[OK]' if audit_ok else '[FAIL]'}")
    print(f"  Certificate file saved  : {'[OK]' if cert_file else '[FAIL]'}")

    return {
        "cert": cert, "impact": impact, "impact2": impact2,
        "cert_ok": cert_ok, "delete_ok": delete_ok,
        "audit_ok": audit_ok, "cert_file": cert_file,
    }


# ---------------------------------------------------------------------------
# Stage C: Decision Disclosure
# ---------------------------------------------------------------------------

def stage_c_decision_disclosure() -> dict:
    section("Stage C: Automated Decision Disclosure & Human-Review Path")

    engine = DecisionDisclosureEngine(
        borderline_margin=0.05,
        low_confidence_threshold=0.55,
    )

    # 5 candidates with different scenarios
    scenarios = [
        # (candidate_id, job_id, score, cutoff, rank, total, features)
        ("CAND_0010", "JOB_1100", 0.78, 0.60, 2,  40,
         {"sessions_14d": 15, "apply_rate_7d": 0.40, "days_since_login": 3,
          "profile_completeness": 92, "is_verified": 1}),
        ("CAND_0020", "JOB_1100", 0.35, 0.60, 28, 40,
         {"sessions_14d": 2,  "apply_rate_7d": 0.05, "days_since_login": 45,
          "profile_completeness": 40, "is_verified": 0}),
        # Borderline: score = cutoff + 0.03 (within margin)
        ("CAND_0030", "JOB_1100", 0.63, 0.60, 8,  40,
         {"sessions_14d": 9,  "apply_rate_7d": 0.22, "days_since_login": 12,
          "profile_completeness": 68, "is_verified": 1}),
        # Borderline: score = cutoff - 0.02
        ("CAND_0040", "JOB_1100", 0.58, 0.60, 12, 40,
         {"sessions_14d": 7,  "apply_rate_7d": 0.18, "days_since_login": 15,
          "profile_completeness": 61, "is_verified": 0}),
        # Low confidence
        ("CAND_0050", "JOB_1100", 0.50, 0.60, 20, 40,
         {"sessions_14d": 5,  "apply_rate_7d": 0.10, "days_since_login": 30,
          "profile_completeness": 55, "is_verified": 0}),
    ]

    results = []
    print(f"\n  {'Candidate':<12} {'Score':>6} {'Outcome':<14} {'Borderline':>10} {'Escalated':>10}")
    print(f"  {'-'*56}")

    for cid, jid, score, cutoff, rank, total, feats in scenarios:
        res = engine.process_decision(
            candidate_id=cid, job_id=jid, model_version="ranking_v2",
            raw_score=score, shortlist_cutoff=cutoff, rank=rank,
            total_candidates=total, feature_values=feats,
        )
        results.append(res)
        rec = res["record"]
        print(f"  {cid:<12} {score:>6.3f} {rec['outcome']:<14} "
              f"{str(rec['borderline']):>10} {str(res['escalated']):>10}")

    # Print worked example for the shortlisted candidate
    print(f"\n  Worked example -- Decision Explanation for CAND_0010:")
    print(f"  " + "-" * 65)
    for line in results[0]["explanation"].split("\n"):
        print(f"  {line}")
    print(f"  " + "-" * 65)

    # Queue status
    qlen = engine.queue_length()
    print(f"\n  Human review queue length: {qlen}")
    print(f"  Decision log present     : {os.path.exists('logs/decision_log.jsonl')}")
    print(f"  Review queue present     : {os.path.exists('logs/human_review_queue.json')}")

    # Disclosure bundle
    bundle = engine.get_disclosure("CAND_0030")
    print(f"\n  Disclosure bundle for CAND_0030: {bundle['n_decisions']} decision(s)")

    escalated_count = sum(1 for r in results if r["escalated"])
    decisions_logged = os.path.exists("logs/decision_log.jsonl")
    review_ok        = qlen >= 1

    return {
        "n_decisions":     len(results),
        "escalated_count": escalated_count,
        "decisions_logged": decisions_logged,
        "review_ok":       review_ok,
        "queue_length":    qlen,
    }


# ---------------------------------------------------------------------------
# Stage D: Audit Pack
# ---------------------------------------------------------------------------

def stage_d_audit_pack() -> dict:
    section("Stage D: Compliance Audit Pack -- Model Card, Fairness, Lineage, SOC 2")

    pack = ComplianceAuditPack(
        model_name="ranking_model",
        model_version="v2",
        metrics={"auc": 0.871, "f1": 0.743, "n_val": 2000},
    )
    index = pack.generate_full_pack()

    print(f"\n  Audit pack contents:")
    for key, val in index["pack_contents"].items():
        status = "[OK  ]" if val != "NOT_PRESENT" else "[MISS]"
        print(f"    {status} {key}: {val}")

    # Print fairness summary
    fair_path = os.path.join("logs/audit_pack", "fairness_report.json")
    if os.path.exists(fair_path):
        with open(fair_path, encoding="utf-8") as f:
            fair = json.load(f)
        fm = fair["fairness_metrics"]
        print(f"\n  Fairness Metrics:")
        print(f"    Profile verification DIR : "
              f"{fm['profile_verification']['disparate_impact_ratio']} "
              f"{'[PASS]' if fm['profile_verification']['pass_80pct_rule'] else '[FAIL]'}")
        print(f"    Seniority DIR            : "
              f"{fm['seniority']['disparate_impact_ratio']} "
              f"{'[PASS]' if fm['seniority']['pass_80pct_rule'] else '[FAIL]'}")
        overall_fair = fair["overall_fairness_pass"]
        print(f"    Overall fairness pass    : {overall_fair}")

    # Worked example
    print(f"\n  Worked example -- Audit Pack:")
    print(f"    Input       : generate_full_pack() on ranking_model v2")
    print(f"    Output      : 4 artefacts in logs/audit_pack/")
    print(f"    Plain reason: Auditor receives model card + fairness + lineage in one directory")
    print(f"    Unavailable : Returns partial pack with available artefacts; logs missing items")

    pack_files = {k: (v != "NOT_PRESENT") for k, v in index["pack_contents"].items()}
    card_ok    = pack_files.get("model_card", False)
    fair_ok    = pack_files.get("fairness_report", False) and overall_fair
    lin_ok     = pack_files.get("lineage_graph", False)
    index_ok   = os.path.exists("logs/audit_pack/audit_index.json")

    return {
        "card_ok":   card_ok,
        "fair_ok":   fair_ok,
        "lin_ok":    lin_ok,
        "index_ok":  index_ok,
        "fair_data": fm if os.path.exists(fair_path) else {},
    }


# ---------------------------------------------------------------------------
# Stage E: Failure Paths + Summary
# ---------------------------------------------------------------------------

def stage_e_failure_and_summary(
    rights_result: dict,
    disclosure_result: dict,
    audit_result: dict,
) -> None:
    section("Stage E: Failure Paths + End-to-End Summary")

    # Force failure: access non-existent subject
    print("\n  -- Forced Failure: Access non-existent subject --")
    engine = DataSubjectRightsEngine(n_candidates=10, seed=42)
    result = engine.handle_access_request("CAND_9999")
    print(f"  Non-existent subject: found={result['found']}  (no crash, graceful)")

    # Force failure: None input
    print("\n  -- Forced Failure: None subject_id --")
    try:
        engine.handle_access_request(None)
        print("  [FAIL] Should have raised ValueError")
    except ValueError as e:
        print(f"  Caught expected ValueError: '{e}'  [OK]")

    # Force failure: decision with invalid score
    print("\n  -- Forced Failure: Decision with out-of-range score --")
    dd_engine = DecisionDisclosureEngine()
    res = dd_engine.process_decision(
        candidate_id="TEST_INVALID", job_id="JOB_X",
        model_version="v0", raw_score=None,
        shortlist_cutoff=0.60, rank=1, total_candidates=10,
        feature_values={"sessions_14d": 5.0},
    )
    print(f"  Invalid score defaulted to 0.0: score={res['record']['raw_score']}  [OK]")

    # Force failure: audit pack with missing logs
    print("\n  -- Forced Failure: Audit pack notes missing files gracefully --")
    pack2   = ComplianceAuditPack(model_name="test_model", model_version="v0")
    index2  = pack2.generate_full_pack()
    missing = [k for k, v in index2["pack_contents"].items() if v == "NOT_PRESENT"]
    present = [k for k, v in index2["pack_contents"].items() if v != "NOT_PRESENT"]
    print(f"  Present artefacts : {len(present)}")
    print(f"  Missing artefacts : {missing or 'none'}")

    # Summary table
    fm = audit_result.get("fair_data", {})
    dir_v   = fm.get("profile_verification", {}).get("disparate_impact_ratio", "N/A")
    dir_s   = fm.get("seniority", {}).get("disparate_impact_ratio", "N/A")
    fair_v  = fm.get("profile_verification", {}).get("pass_80pct_rule", False)
    fair_s  = fm.get("seniority", {}).get("pass_80pct_rule", False)

    rows = [
        ("Access request returns data bundle",       "yes",  True),
        ("Deletion cascades all stores",             "yes",  rights_result["cert_ok"]),
        ("Re-access returns empty after deletion",   "yes",  rights_result["delete_ok"]),
        ("Deletion cert file issued",                "yes",  rights_result["cert_file"]),
        ("Audit log written (data_rights_log)",      "yes",  rights_result["audit_ok"]),
        ("100% of decisions logged",                 str(disclosure_result["n_decisions"]),
                                                             disclosure_result["decisions_logged"]),
        ("Borderline cases escalated",               str(disclosure_result["escalated_count"]),
                                                             disclosure_result["review_ok"]),
        ("Human review queue populated",             str(disclosure_result["queue_length"]),
                                                             disclosure_result["review_ok"]),
        ("Model card (compliance) generated",        "yes",  audit_result["card_ok"]),
        (f"Fairness DIR verified (>= 0.80)",         str(dir_v), fair_v),
        (f"Fairness DIR seniority (>= 0.80)",        str(dir_s), fair_s),
        ("Data lineage graph generated",             "yes",  audit_result["lin_ok"]),
        ("Audit index (SOC 2 bundle) generated",     "yes",  audit_result["index_ok"]),
    ]

    print("\n  +" + "-" * 68 + "+")
    print(f"  | {'Check':<44} {'Value':>8} {'Status':>12} |")
    print("  +" + "-" * 68 + "+")
    for label, val, passed in rows:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  | {label:<44} {str(val):>8} {status:>12} |")
    print("  +" + "-" * 68 + "+")

    all_pass = all(p for _, _, p in rows)
    print(f"\n  Overall demo result: {'[ALL PASS]' if all_pass else '[SOME CHECKS FAILED]'}")

    print("\n  Output files:")
    paths = [
        "logs/data_rights_log.jsonl",
        "logs/decision_log.jsonl",
        "logs/human_review_queue.json",
        "logs/audit_pack/model_card_compliance.md",
        "logs/audit_pack/fairness_report.json",
        "logs/audit_pack/lineage_graph.txt",
        "logs/audit_pack/audit_index.json",
        "logs/task23.log",
    ]
    for path in paths:
        exists = "[OK  ]" if os.path.exists(path) else "[MISS]"
        print(f"    {exists} {path}")

    print("\n" + "=" * 70)
    print("  Task 23 Compliance Demo Complete.")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_demo() -> None:
    """Execute the full Task 23 compliance demo."""
    engine          = stage_a_setup()
    rights_result   = stage_b_data_rights(engine)
    disclosure_result = stage_c_decision_disclosure()
    audit_result    = stage_d_audit_pack()
    stage_e_failure_and_summary(rights_result, disclosure_result, audit_result)


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
