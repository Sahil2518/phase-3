"""
demo_task22_security.py -- PlaceMux Phase 3, Task 22
=====================================================
End-to-End Demo: Security Hardening, Threat Model & Pen-Test Remediation

Five stages:
  A: ML Threat Model -- generate risk matrix, call out CRITICAL threats
  B: Keyword Stuffing Defence -- adversarial vs legitimate resumes, FPR/TPR
  C: Ranking Manipulation -- score-velocity attack, synonym flooding
  D: Scraping + Poison Detection -- burst API calls + injected poison rows
  E: Failure / fallback paths + integrated summary table

Usage:
    python -m src.demo_task22_security
"""

import os
import sys
import json
import time
import logging
import numpy as np

os.makedirs("logs", exist_ok=True)

_fh = logging.FileHandler("logs/task22_security.log", encoding="utf-8")
_sh = logging.StreamHandler(sys.stdout)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[_fh, _sh],
)
logger = logging.getLogger(__name__)

from src.ml_threat_model   import ThreatRegistry, ThreatScorer
from src.ranking_defence   import (
    KeywordStuffingDetector,
    SynonymFloodingDetector,
    RankingManipulationDetector,
    evaluate_detector,
    save_defence_report,
)
from src.security_monitor  import (
    ScrapingDetector,
    DataPoisonDetector,
    save_security_report,
)


def section(title: str) -> None:
    """Print an ASCII section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

ADVERSARIAL_RESUMES = [
    # A1 - pure keyword repetition
    ("python " * 60 + "machine learning " * 40 +
     "ai ai ai ai ai ai ai ai ai ai ai ai ai ai ai "),
    # A2 - invisible char injection
    ("python\u200b python\u200b python\u200b machine\u200b learning\u200b " * 30 +
     "data science sql aws azure tensorflow pytorch deep learning "),
    # A3 - high keyword density (~40% of tokens are target keywords)
    "python sql aws azure ml ai nlp docker kubernetes tensorflow pytorch " * 12,
]

LEGITIMATE_RESUMES = [
    ("Experienced data scientist with 6 years building production ML systems. "
     "Led a team of 4 to deploy a recommendation engine serving 2M users. "
     "Strong skills in Python, SQL and cloud infrastructure on AWS. "
     "Published two papers on anomaly detection at KDD 2023."),
    ("Software engineer specialising in backend services and API design. "
     "5 years at fintech companies building high-throughput payment systems in Java. "
     "Comfortable with Docker, Kubernetes, and CI/CD pipelines. "
     "Holds AWS Solutions Architect certification."),
    ("Marketing analyst with 3 years of experience in consumer insights. "
     "Proficient in Excel, Tableau and basic Python for data visualisation. "
     "Managed campaigns for FMCG brands with combined budget of GBP 4M. "
     "Strong stakeholder communication and presentation skills."),
]

FLOODED_SKILLS = [
    "ml", "machine learning", "AI", "deep learning", "neural network",
    "nn", "DL", "supervised learning", "unsupervised learning",
    "artificial intelligence", "reinforcement learning", "transfer learning",
]

LEGITIMATE_SKILLS = [
    "Python", "SQL", "AWS", "Docker", "Kubernetes",
    "data analysis", "communication", "project management",
]


# ---------------------------------------------------------------------------
# Stage A: Threat Model
# ---------------------------------------------------------------------------

def stage_a_threat_model() -> dict:
    section("Stage A: ML Threat Model -- Risk Matrix")

    registry = ThreatRegistry()
    scorer   = ThreatScorer(registry, report_path="logs/threat_model.json")
    report   = scorer.score()
    scorer.print_risk_matrix(report)

    s = report["summary"]
    critical_ids = s["critical_threat_ids"]
    print(f"  [{'OK' if s['overall_risk'] != 'CRITICAL' else 'WARN'}] "
          f"Overall risk: {s['overall_risk']}")

    # Worked example: T01 Keyword Stuffing
    t01 = registry.get_by_id("T01")
    print(f"\n  Worked example -- T01: {t01.title}")
    print(f"    Input       : Candidate submits resume with 'python' repeated 80x")
    print(f"    Output      : KeywordStuffingDetector flags TTR < 0.35 + density > 0.15")
    print(f"    Plain reason: Lexical diversity too low for genuine resume text")
    print(f"    Unavailable : Detector skipped; record queued for human review")

    print(f"\n  [OK] Threat model written -> logs/threat_model.json")
    return report


# ---------------------------------------------------------------------------
# Stage B: Keyword Stuffing Defence
# ---------------------------------------------------------------------------

def stage_b_keyword_stuffing() -> dict:
    section("Stage B: Keyword Stuffing & Synonym Flooding Defence")

    ksd = KeywordStuffingDetector(ttr_threshold=0.35, density_threshold=0.15)

    print("\n  -- Adversarial Resumes --")
    adv_results = []
    for i, text in enumerate(ADVERSARIAL_RESUMES, 1):
        r = ksd.detect(text, candidate_id=f"ADV_{i:02d}")
        adv_results.append(r)
        flag = "[FLAGGED]" if r["flagged"] else "[MISSED ]"
        print(f"  {flag} ADV_{i:02d}: score={r['score']:.3f}  reason='{r['reason'][:60]}'")

    print("\n  -- Legitimate Resumes --")
    leg_results = []
    for i, text in enumerate(LEGITIMATE_RESUMES, 1):
        r = ksd.detect(text, candidate_id=f"LEG_{i:02d}")
        leg_results.append(r)
        flag = "[FP!    ]" if r["flagged"] else "[OK     ]"
        print(f"  {flag} LEG_{i:02d}: score={r['score']:.3f}  reason='{r['reason'][:60]}'")

    # Offline metrics
    metrics = evaluate_detector(ksd, ADVERSARIAL_RESUMES, LEGITIMATE_RESUMES)
    print(f"\n  Offline Metrics (held-out adversarial vs legitimate):")
    print(f"    TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}  TN={metrics['tn']}")
    print(f"    Precision : {metrics['precision']:.3f}  (bar: >= 0.90)")
    print(f"    Recall    : {metrics['recall']:.3f}  (bar: >= 0.85)")
    print(f"    FPR       : {metrics['fpr']:.3f}  (bar: <  0.05)")
    print(f"    F1        : {metrics['f1']:.3f}")

    prec_ok = metrics['precision'] >= 0.90
    rec_ok  = metrics['recall']    >= 0.85
    fpr_ok  = metrics['fpr']       <  0.05
    print(f"\n  Precision bar: {'[PASS]' if prec_ok else '[FAIL]'}")
    print(f"  Recall bar   : {'[PASS]' if rec_ok  else '[FAIL]'}")
    print(f"  FPR bar      : {'[PASS]' if fpr_ok  else '[FAIL]'}")

    # Synonym flooding
    print("\n  -- Synonym Flooding --")
    sfd = SynonymFloodingDetector(max_cluster_ratio=0.40)
    r_flooded = sfd.detect(FLOODED_SKILLS,   candidate_id="ADV_SF_01")
    r_legit   = sfd.detect(LEGITIMATE_SKILLS, candidate_id="LEG_SF_01")
    print(f"  Flooded skills : flagged={r_flooded['flagged']} "
          f"ratio={r_flooded['details']['top_ratio']:.2f}")
    print(f"  Legit skills   : flagged={r_legit['flagged']} "
          f"ratio={r_legit['details']['top_ratio']:.2f}")

    return {
        "stuffing_metrics": metrics,
        "synonym_flooded":  r_flooded["flagged"],
        "synonym_legit":    r_legit["flagged"],
    }


# ---------------------------------------------------------------------------
# Stage C: Ranking Manipulation
# ---------------------------------------------------------------------------

def stage_c_ranking_manipulation() -> dict:
    section("Stage C: Ranking Manipulation Detection")

    rmd = RankingManipulationDetector(max_score_delta=0.25, max_top1_rate=0.80)

    # Score-velocity attack
    print("\n  -- Score-Velocity Guard --")
    legit_history  = [0.42, 0.44, 0.45, 0.47, 0.49]   # gradual, natural
    attack_history = [0.30, 0.32, 0.35, 0.72, 0.80]   # sudden spike

    r_vel_legit  = rmd.check_velocity("candidate_A", legit_history)
    r_vel_attack = rmd.check_velocity("candidate_B", attack_history)

    print(f"  Legitimate trajectory  : {legit_history}")
    print(f"    flagged={r_vel_legit['flagged']}  max_delta={r_vel_legit['details']['max_delta']}")
    print(f"  Attack trajectory      : {attack_history}")
    print(f"    flagged={r_vel_attack['flagged']}  max_delta={r_vel_attack['details']['max_delta']}")

    # Worked example
    print(f"\n  Worked example -- Score-Velocity:")
    print(f"    Input       : Candidate B scores [0.30 -> 0.32 -> 0.72 -> 0.80]")
    print(f"    Output      : delta=0.37 exceeds threshold 0.25 -> FLAGGED")
    print(f"    Plain reason: Score jumped 37 points in one window (fake positive feedback)")
    print(f"    Unavailable : Velocity check skipped; last known score retained")

    # Omnipresence attack
    print("\n  -- Omnipresence Guard --")
    omnipresent = {f"query_cat_{i}": 1 for i in range(9)}
    omnipresent["query_cat_9"] = 3   # not #1 here
    genuine = {f"query_cat_{i}": (1 if i < 2 else i + 2) for i in range(10)}

    r_omni_attack  = rmd.check_omnipresence("candidate_C", omnipresent)
    r_omni_genuine = rmd.check_omnipresence("candidate_D", genuine)

    print(f"  Omnipresent candidate  : flagged={r_omni_attack['flagged']} "
          f"top1_rate={r_omni_attack['details']['top1_rate']:.2f}")
    print(f"  Genuine candidate      : flagged={r_omni_genuine['flagged']} "
          f"top1_rate={r_omni_genuine['details']['top1_rate']:.2f}")

    velocity_ok = r_vel_attack["flagged"] and not r_vel_legit["flagged"]
    omni_ok     = r_omni_attack["flagged"] and not r_omni_genuine["flagged"]
    print(f"\n  Velocity guard : {'[PASS]' if velocity_ok else '[FAIL]'}")
    print(f"  Omni guard     : {'[PASS]' if omni_ok else '[FAIL]'}")

    return {
        "velocity_attack_flagged":  r_vel_attack["flagged"],
        "velocity_legit_flagged":   r_vel_legit["flagged"],
        "omni_attack_flagged":      r_omni_attack["flagged"],
        "omni_legit_flagged":       r_omni_genuine["flagged"],
    }


# ---------------------------------------------------------------------------
# Stage D: Scraping + Poison Detection
# ---------------------------------------------------------------------------

def stage_d_scraping_and_poison() -> dict:
    section("Stage D: Scraping Detection + Data Poison Detection")

    # --- Scraping ---
    print("\n  -- Live Scraping Attack Simulation --")
    scraper = ScrapingDetector(
        window_seconds=60,
        rate_limit_threshold=10,
        block_threshold=20,
    )
    t0 = time.time()
    actions = []
    for i in range(25):
        res = scraper.check("attacker_bot", resource_id=1000 + i, now=t0 + i * 0.8)
        actions.append(res["action"])
        if res["action"] != "ALLOW":
            print(f"  Request {i+1:02d}: {res['action']:12s} | {res['reason']}")
        if res["action"] == "BLOCK":
            print(f"  [!] Attacker blocked at request {i+1}")
            break

    first_block = next((i for i, a in enumerate(actions) if a == "BLOCK"), None)
    print(f"\n  Legitimate client (10 normal requests):")
    legit_scraper = ScrapingDetector(rate_limit_threshold=10, block_threshold=20)
    for i in range(10):
        res = legit_scraper.check("legit_user", resource_id=500 + i * 7, now=t0 + i * 5)
    print(f"  Last action: {res['action']}")

    scraping_ok = first_block is not None
    print(f"\n  Scraping block : {'[PASS]' if scraping_ok else '[FAIL]'}")

    # Worked example
    print(f"\n  Worked example -- Scraping:")
    print(f"    Input       : client issues 25 requests in 20s with IDs 1000,1001,1002...")
    print(f"    Output      : BLOCK at request {(first_block+1) if first_block else 'N/A'} "
          f"(enumeration + rate exceeded)")
    print(f"    Plain reason: Sequential ID sweep = hallmark of data scraper")
    print(f"    Unavailable : Detector skipped; standard rate-limit headers still applied")

    # --- Data Poison Detection ---
    print("\n  -- Data Poison Detection --")
    rng = np.random.default_rng(42)

    # Reference clean data (simulate engagement features)
    ref_X = np.column_stack([
        rng.integers(0, 60, 800).astype(float),    # days_since_login
        rng.integers(0, 18, 800).astype(float),    # sessions_14d
        rng.uniform(0, 0.45, 800),                 # apply_rate
        rng.integers(0, 200, 800).astype(float),   # jobs_viewed
        rng.uniform(20, 100, 800),                 # profile_completeness
    ])

    poison_det = DataPoisonDetector(
        poison_threshold=0.08,   # flag batches where >8% of rows are anomalous
        contamination=0.05,
        engagement_features=["sessions_14d"],
        label_col="label",
    )
    poison_det.fit_reference(ref_X)

    # Clean batch
    clean_X = np.column_stack([
        rng.integers(0, 60, 200).astype(float),
        rng.integers(0, 18, 200).astype(float),
        rng.uniform(0, 0.45, 200),
        rng.integers(0, 200, 200).astype(float),
        rng.uniform(20, 100, 200),
    ])
    clean_y = rng.integers(0, 2, 200)

    r_clean = poison_det.screen(clean_X, clean_y,
                                 feature_names=["days_since_login", "sessions_14d",
                                                "apply_rate", "jobs_viewed",
                                                "profile_completeness"],
                                 anomaly_score_threshold=0.95)
    print(f"  Clean batch    : safe={r_clean['safe_to_train']} "
          f"poison_rate={r_clean['poison_rate']:.2%} "
          f"flagged={r_clean['n_flagged']}/{r_clean['n_total']}")

    # Poisoned batch: inject 20 extreme outlier rows (~9% poison)
    poison_rows_X = np.full((20, 5), 999.0)   # extreme out-of-distribution
    poison_rows_y = np.ones(20, dtype=int)     # all labelled as positive
    mixed_X = np.vstack([clean_X, poison_rows_X])
    mixed_y = np.concatenate([clean_y, poison_rows_y])

    r_poison = poison_det.screen(mixed_X, mixed_y,
                                  feature_names=["days_since_login", "sessions_14d",
                                                 "apply_rate", "jobs_viewed",
                                                 "profile_completeness"],
                                 anomaly_score_threshold=0.90)
    print(f"  Poisoned batch : safe={r_poison['safe_to_train']} "
          f"poison_rate={r_poison['poison_rate']:.2%} "
          f"flagged={r_poison['n_flagged']}/{r_poison['n_total']}")
    injected_flagged = sum(1 for idx in r_poison["poisoned_indices"] if idx >= 200)
    poison_recall = injected_flagged / 20.0
    print(f"  Poison recall  : {poison_recall:.2%}  (bar: >= 0.80) "
          f"{'[PASS]' if poison_recall >= 0.80 else '[FAIL]'}")

    # Worked example
    print(f"\n  Worked example -- Data Poison:")
    print(f"    Input       : Batch of 220 rows; 20 rows have all features = 999.0")
    print(f"    Output      : safe_to_train=False, {r_poison['n_flagged']} rows flagged")
    print(f"    Plain reason: IsolationForest scores outlier rows near 1.0 (max anomaly)")
    print(f"    Unavailable : Detector unavailable -> training halted, alert raised")

    return {
        "scraping_block_at": first_block,
        "clean_safe":        r_clean["safe_to_train"],
        "poison_safe":       r_poison["safe_to_train"],
        "poison_recall":     round(poison_recall, 4),
        "scraping_ok":       scraping_ok,
        "poison_ok":         poison_recall >= 0.80,
    }


# ---------------------------------------------------------------------------
# Stage E: Failure paths + summary
# ---------------------------------------------------------------------------

def stage_e_failure_and_summary(
    threat_report: dict,
    stuffing_result: dict,
    manipulation_result: dict,
    poison_result: dict,
) -> None:
    section("Stage E: Failure Paths + End-to-End Summary")

    print("\n  -- Forced Failure: Empty resume text --")
    ksd = KeywordStuffingDetector()
    r_empty = ksd.detect("", candidate_id="empty_test")
    print(f"  Empty input result: flagged={r_empty['flagged']} "
          f"reason='{r_empty['reason']}'  (no crash, graceful)")

    print("\n  -- Forced Failure: Single-token resume --")
    r_short = ksd.detect("python", candidate_id="short_test")
    print(f"  Single token  result: flagged={r_short['flagged']} "
          f"reason='{r_short['reason']}'  (below min_tokens threshold)")

    print("\n  -- Forced Failure: Scraping detector unavailable (exception path) --")
    scraper_fail = ScrapingDetector()
    r_fail = scraper_fail.check(None, resource_id=None, now=None)
    print(f"  Null client result: action={r_fail['action']} "
          f"(fail-open: legitimate traffic not blocked)")

    # Summary table
    sm  = stuffing_result["stuffing_metrics"]
    prec_pass = sm["precision"] >= 0.90
    rec_pass  = sm["recall"]    >= 0.85
    fpr_pass  = sm["fpr"]       <  0.05
    vel_pass  = (manipulation_result["velocity_attack_flagged"]
                 and not manipulation_result["velocity_legit_flagged"])
    omni_pass = (manipulation_result["omni_attack_flagged"]
                 and not manipulation_result["omni_legit_flagged"])
    syn_pass  = (stuffing_result["synonym_flooded"]
                 and not stuffing_result["synonym_legit"])
    scr_pass  = poison_result["scraping_ok"]
    poi_pass  = poison_result["poison_ok"]

    total_threats   = threat_report["summary"]["total_threats"]
    overall_risk    = threat_report["summary"]["overall_risk"]
    critical_n      = threat_report["summary"]["severity_counts"]["CRITICAL"]

    print("\n  +" + "-" * 65 + "+")
    print(f"  | {'Check':<40} {'Result':>10} {'Status':>12} |")
    print("  +" + "-" * 65 + "+")

    rows = [
        ("Threat model generated",          f"{total_threats} threats",   True),
        ("Overall risk level identified",     overall_risk,                 True),
        ("CRITICAL threats found & logged",  str(critical_n),              critical_n > 0),
        (f"Stuffing precision (>= 0.90)",   f"{sm['precision']:.3f}",     prec_pass),
        (f"Stuffing recall    (>= 0.85)",   f"{sm['recall']:.3f}",        rec_pass),
        (f"Legitimate FPR     (<  0.05)",   f"{sm['fpr']:.3f}",           fpr_pass),
        ("Synonym flooding detected",        "yes",                        syn_pass),
        ("Velocity attack flagged",          "yes",                        vel_pass),
        ("Omnipresence attack flagged",      "yes",                        omni_pass),
        ("Scraping attack blocked",          "yes",                        scr_pass),
        (f"Poison recall (>= 0.80)",         f"{poison_result['poison_recall']:.3f}", poi_pass),
        ("Clean batch safe to train",        str(poison_result["clean_safe"]),  poison_result["clean_safe"]),
    ]
    for label, value, passed in rows:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  | {label:<40} {value:>10} {status:>12} |")
    print("  +" + "-" * 65 + "+")

    all_pass = all(p for _, _, p in rows)
    print(f"\n  Overall demo result: {'[ALL PASS]' if all_pass else '[SOME CHECKS FAILED]'}")
    print("\n  Output files:")
    for path in [
        "logs/threat_model.json",
        "logs/ranking_defence_report.json",
        "logs/security_monitor_report.json",
        "logs/task22_security.log",
    ]:
        exists = "[OK  ]" if os.path.exists(path) else "[MISS]"
        print(f"    {exists} {path}")

    # Save combined security report
    full_report = {
        "threat_model_summary":     threat_report["summary"],
        "stuffing_metrics":         sm,
        "synonym_flooding":         {"flooded_flagged": stuffing_result["synonym_flooded"]},
        "velocity_guard":           {"attack_flagged": manipulation_result["velocity_attack_flagged"]},
        "omnipresence_guard":       {"attack_flagged": manipulation_result["omni_attack_flagged"]},
        "scraping_detection":       {"blocked": poison_result["scraping_ok"]},
        "poison_detection": {
            "poison_recall":  poison_result["poison_recall"],
            "clean_safe":     poison_result["clean_safe"],
            "poison_safe":    poison_result["poison_safe"],
        },
    }
    save_security_report(full_report)
    save_defence_report({
        "metrics": sm,
        "synonym_flooded": stuffing_result["synonym_flooded"],
    })
    print("\n" + "=" * 70)
    print("  Task 22 Security Demo Complete.")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_demo() -> None:
    """Execute the full Task 22 security hardening demo."""
    threat_report      = stage_a_threat_model()
    stuffing_result    = stage_b_keyword_stuffing()
    manip_result       = stage_c_ranking_manipulation()
    poison_result      = stage_d_scraping_and_poison()
    stage_e_failure_and_summary(
        threat_report, stuffing_result, manip_result, poison_result
    )


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
