"""
ml_threat_model.py -- PlaceMux Phase 3, Task 22
================================================
ML Threat Model for the PlaceMux Intelligence Layer.

Design rationale
----------------
STRIDE-inspired threat registry covering every attack surface in the
PlaceMux ML pipeline (data ingestion -> feature engineering -> training ->
serving -> API layer).  Each threat is rated by impact (H/M/L) and
likelihood (H/M/L), producing a residual risk score 1-9.  A ThreatScorer
flags CRITICAL threats and writes logs/threat_model.json.

Usage
-----
    from src.ml_threat_model import ThreatRegistry, ThreatScorer
    registry = ThreatRegistry()
    scorer   = ThreatScorer(registry)
    report   = scorer.score()
"""

import os
import sys
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import List, Dict, Optional

os.makedirs("logs", exist_ok=True)
logger = logging.getLogger(__name__)

IMPACT_WEIGHT     = {"H": 3, "M": 2, "L": 1}
LIKELIHOOD_WEIGHT = {"H": 3, "M": 2, "L": 1}


@dataclass
class Threat:
    """
    Single threat entry in the ML threat registry.

    Parameters
    ----------
    threat_id     : str   e.g. 'T01'
    category      : str   STRIDE bucket
    title         : str   Short human-readable name
    description   : str   What the attacker does
    attack_vector : str   Entry point into the system
    attacker_goal : str   What they gain if the attack succeeds
    impact        : str   'H', 'M', or 'L'
    likelihood    : str   'H', 'M', or 'L'
    mitigations   : List[str]  Controls already deployed
    """

    threat_id:     str
    category:      str
    title:         str
    description:   str
    attack_vector: str
    attacker_goal: str
    impact:        str
    likelihood:    str
    mitigations:   List[str] = field(default_factory=list)
    residual_risk: int = field(init=False, default=0)
    severity:      str = field(init=False, default="")

    def compute_risk(self) -> None:
        """
        Compute residual risk score (1-9) and severity label.

        residual_risk = impact_weight x likelihood_weight.
        severity: 9->CRITICAL, >=7->HIGH, >=4->MEDIUM, else LOW.
        """
        iw = IMPACT_WEIGHT.get(self.impact, 1)
        lw = LIKELIHOOD_WEIGHT.get(self.likelihood, 1)
        self.residual_risk = iw * lw
        if self.residual_risk >= 9:
            self.severity = "CRITICAL"
        elif self.residual_risk >= 7:
            self.severity = "HIGH"
        elif self.residual_risk >= 4:
            self.severity = "MEDIUM"
        else:
            self.severity = "LOW"


class ThreatRegistry:
    """
    Catalogue of 12 attack vectors for the PlaceMux ML system.

    Threats cover: keyword stuffing, invisible chars, synonym flooding,
    role inflation, training data poisoning, model extraction, bulk scraping,
    prompt injection, feedback loop gaming, omnipresence attack, cross-tenant
    leakage, and timing-channel inference.
    """

    def __init__(self) -> None:
        """Initialise and populate all 12 threats."""
        self.threats: List[Threat] = []
        self._populate()

    def _populate(self) -> None:
        """Define and register all 12 threat entries."""
        raw = [
            dict(
                threat_id="T01", category="Tampering",
                title="Keyword Stuffing",
                description=(
                    "Candidate inflates resume with hundreds of repetitions of "
                    "high-value keywords to maximise TF-IDF / BM25 match scores."
                ),
                attack_vector="Resume text field via candidate profile API",
                attacker_goal="Rank higher than genuinely qualified candidates",
                impact="H", likelihood="H",
                mitigations=[
                    "KeywordStuffingDetector: TTR + keyword density threshold",
                    "Token repetition cap enforced at ingestion",
                ],
            ),
            dict(
                threat_id="T02", category="Tampering",
                title="Invisible Character Injection",
                description=(
                    "Attacker embeds zero-width spaces (U+200B), soft hyphens "
                    "(U+00AD), or Unicode homoglyphs to hide repeated keywords "
                    "from human reviewers while keeping them machine-readable."
                ),
                attack_vector="Resume text field (Unicode-aware)",
                attacker_goal="Bypass human review while inflating machine keyword count",
                impact="H", likelihood="M",
                mitigations=[
                    "KeywordStuffingDetector: scans for zero-width / control chars",
                    "Text normalisation strips non-printable Unicode at ingest",
                ],
            ),
            dict(
                threat_id="T03", category="Tampering",
                title="Synonym Flooding / Adversarial Skill Expansion",
                description=(
                    "Candidate lists 30+ near-synonyms of the same skill "
                    "(e.g. 'ML, machine learning, AI, deep learning...') to inflate "
                    "skill-match breadth without genuine competence."
                ),
                attack_vector="Skills section of candidate profile",
                attacker_goal="Appear broadly skilled; match more job descriptions",
                impact="M", likelihood="H",
                mitigations=[
                    "SynonymFloodingDetector: near-synonym cluster density check",
                    "Skill ontology canonicalisation collapses synonyms pre-scoring",
                ],
            ),
            dict(
                threat_id="T04", category="Spoofing",
                title="Role Title / Seniority Inflation",
                description=(
                    "Candidate self-reports inflated titles or fabricates years "
                    "of experience to pass seniority-based ranking filters."
                ),
                attack_vector="Work experience structured fields",
                attacker_goal="Pass seniority filters; reach senior-role shortlists",
                impact="M", likelihood="M",
                mitigations=[
                    "Cross-reference company size signals from external data",
                    "Seniority outlier detector: flag >2-sigma title vs company size",
                ],
            ),
            dict(
                threat_id="T05", category="Tampering",
                title="Training Data Poisoning",
                description=(
                    "Attacker submits batches of synthetic or manipulated interaction "
                    "events (fake clicks, applies, recruiter feedback) to shift the "
                    "model's learned ranking function."
                ),
                attack_vector="Interaction event pipeline (click / apply / feedback APIs)",
                attacker_goal="Corrupt model so target candidates rank higher permanently",
                impact="H", likelihood="M",
                mitigations=[
                    "DataPoisonDetector: Isolation Forest on training batch statistics",
                    "Label consistency check: flags feature-label contradictions",
                    "Duplicate injection guard: Jaccard near-duplicate threshold",
                    "Training data provenance audit log",
                ],
            ),
            dict(
                threat_id="T06", category="Information Disclosure",
                title="Model Score Extraction / Inversion Attack",
                description=(
                    "Attacker makes thousands of crafted ranking API calls to "
                    "reverse-engineer model feature weights, enabling resumes that "
                    "deterministically score near 1.0."
                ),
                attack_vector="Ranking / scoring API",
                attacker_goal="Extract model internals to craft guaranteed top-rank resumes",
                impact="H", likelihood="M",
                mitigations=[
                    "ScrapingDetector: rate-window + enumeration pattern detection",
                    "Score output rounding / noise injection",
                    "API auth + per-token rate limits",
                ],
            ),
            dict(
                threat_id="T07", category="Information Disclosure",
                title="Bulk Candidate / Job Data Scraping",
                description=(
                    "Competitor systematically iterates candidate_ids or job_ids "
                    "to download the full PlaceMux dataset via the API."
                ),
                attack_vector="Candidate profile and job listing endpoints",
                attacker_goal="Steal proprietary talent and job datasets",
                impact="H", likelihood="H",
                mitigations=[
                    "ScrapingDetector: sequential-ID enumeration + rate-window block",
                    "API rate limiting (60 req/min per token)",
                    "CAPTCHA challenge on anomalous access patterns",
                ],
            ),
            dict(
                threat_id="T08", category="Tampering",
                title="Prompt Injection via Resume / JD Text",
                description=(
                    "If an LLM parses resumes or JDs, attacker embeds adversarial "
                    "instructions in plaintext to hijack LLM reasoning and force a "
                    "favourable structured parse output."
                ),
                attack_vector="Resume free-text fields sent to LLM parsing pipeline",
                attacker_goal="Override LLM output to produce a favourable parse",
                impact="H", likelihood="M",
                mitigations=[
                    "Structural output schema validation (reject freeform overrides)",
                    "Sandboxed LLM calls with output type checking",
                    "Text sanitisation: strip instruction-like patterns before LLM call",
                ],
            ),
            dict(
                threat_id="T09", category="Tampering",
                title="Positive Feedback Loop Injection",
                description=(
                    "Coordinated fake recruiter accounts click/apply to specific "
                    "candidates repeatedly to inflate implicit feedback signals, "
                    "biasing future model retraining cycles."
                ),
                attack_vector="Recruiter click/apply event stream",
                attacker_goal="Game retraining to permanently boost target candidates",
                impact="H", likelihood="M",
                mitigations=[
                    "RankingManipulationDetector: score-velocity guard across sessions",
                    "Recruiter account anomaly scoring",
                    "DataPoisonDetector on each retrain batch",
                ],
            ),
            dict(
                threat_id="T10", category="Tampering",
                title="Ranking Omnipresence Attack",
                description=(
                    "Candidate crafts a profile that scores top-3 across ALL "
                    "query types -- an impossibility for a genuine candidate, "
                    "indicating adversarial profile crafting."
                ),
                attack_vector="Candidate profile feature vector",
                attacker_goal="Appear in every recruiter shortlist regardless of fit",
                impact="M", likelihood="L",
                mitigations=[
                    "RankingManipulationDetector: omnipresence guard (top-rank rate >80%)",
                    "Query-diversity flag: alert when candidate ranks #1 across distinct roles",
                ],
            ),
            dict(
                threat_id="T11", category="Information Disclosure",
                title="Cross-Tenant Signal Leakage",
                description=(
                    "A recruiter from Tenant A can infer candidate data belonging to "
                    "Tenant B if shared ML features leak cross-tenant signals via a "
                    "shared embedding space."
                ),
                attack_vector="Shared ranking model trained on multi-tenant data",
                attacker_goal="Obtain competitor's candidate intelligence",
                impact="H", likelihood="L",
                mitigations=[
                    "Per-tenant feature isolation (Task 18 multi-tenancy)",
                    "Tenant-scoped model inference with input/output audit logs",
                    "Differential privacy noise injection on shared embeddings",
                ],
            ),
            dict(
                threat_id="T12", category="Information Disclosure",
                title="Timing-Channel Score Inference",
                description=(
                    "Attacker measures API response latency to infer whether a "
                    "candidate is shortlisted (fast cached path) vs excluded (slow "
                    "full-ranking path), leaking ranking outcomes as a side channel."
                ),
                attack_vector="Scoring API response time",
                attacker_goal="Learn ranking outcome without seeing the response body",
                impact="L", likelihood="L",
                mitigations=[
                    "Constant-time response padding (random sleep 0-20ms)",
                    "Cache key hashing to prevent cache-timing distinguishability",
                ],
            ),
        ]

        for t in raw:
            threat = Threat(**t)
            threat.compute_risk()
            self.threats.append(threat)

        logger.info(f"ThreatRegistry populated with {len(self.threats)} threats.")

    def get_by_id(self, threat_id: str) -> Optional[Threat]:
        """
        Retrieve a single threat by its ID.

        Parameters
        ----------
        threat_id : str  e.g. 'T01'

        Returns
        -------
        Threat or None
        """
        for t in self.threats:
            if t.threat_id == threat_id:
                return t
        return None

    def get_by_severity(self, severity: str) -> List[Threat]:
        """
        Filter threats by severity label.

        Parameters
        ----------
        severity : str  'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW'

        Returns
        -------
        List[Threat]
        """
        return [t for t in self.threats if t.severity == severity]


class ThreatScorer:
    """
    Scores the PlaceMux ML system against the threat registry, builds a risk
    matrix, and writes a structured JSON report.

    Parameters
    ----------
    registry    : ThreatRegistry
    report_path : str  Path to write the JSON report.
    """

    def __init__(
        self,
        registry: ThreatRegistry,
        report_path: str = "logs/threat_model.json",
    ) -> None:
        """Initialise with registry and output path."""
        self.registry    = registry
        self.report_path = report_path

    def score(self) -> Dict:
        """
        Walk the registry, compute aggregate statistics, and persist the report.

        Returns
        -------
        dict  Full threat model report with summary and all threat entries.
        """
        threats_data = [asdict(t) for t in self.registry.threats]

        severity_counts: Dict[str, int] = {
            "CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0
        }
        critical_threats: List[str] = []

        for t in self.registry.threats:
            severity_counts[t.severity] = severity_counts.get(t.severity, 0) + 1
            if t.severity == "CRITICAL":
                critical_threats.append(t.threat_id)

        if critical_threats:
            overall_risk = "CRITICAL"
        elif severity_counts["HIGH"] > 0:
            overall_risk = "HIGH"
        elif severity_counts["MEDIUM"] > 0:
            overall_risk = "MEDIUM"
        else:
            overall_risk = "LOW"

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "system":   "PlaceMux Intelligence Layer",
            "sprint":   "Phase 3, Sprint E",
            "task":     "Task 22 - Security Hardening",
            "summary": {
                "total_threats":      len(self.registry.threats),
                "severity_counts":    severity_counts,
                "overall_risk":       overall_risk,
                "critical_threat_ids": critical_threats,
                "one_line_bar": (
                    "A candidate cannot game their way to the top, "
                    "and an attacker cannot steal or poison the model."
                ),
            },
            "threats": threats_data,
        }

        try:
            with open(self.report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Threat model report saved: {self.report_path}")
        except Exception as e:
            logger.error(f"Failed to save threat model report: {e}")

        return report

    def print_risk_matrix(self, report: Dict) -> None:
        """
        Print a formatted risk matrix table to stdout.

        Parameters
        ----------
        report : dict  The output of .score().
        """
        print("\n" + "=" * 72)
        print("  PlaceMux ML Threat Model -- Risk Matrix")
        print("=" * 72)
        print(f"  {'ID':<5} {'Severity':<10} {'Risk':>5}  Title")
        print("  " + "-" * 68)

        threats_sorted = sorted(
            self.registry.threats, key=lambda t: t.residual_risk, reverse=True
        )
        for t in threats_sorted:
            flag = " [!!!]" if t.severity == "CRITICAL" else ""
            print(
                f"  {t.threat_id:<5} {t.severity:<10} "
                f"{t.residual_risk:>5}  {t.title}{flag}"
            )

        s = report["summary"]
        print(f"\n  Total: {s['total_threats']}  |  "
              f"CRITICAL: {s['severity_counts']['CRITICAL']}  |  "
              f"HIGH: {s['severity_counts']['HIGH']}  |  "
              f"MEDIUM: {s['severity_counts']['MEDIUM']}  |  "
              f"LOW: {s['severity_counts']['LOW']}")
        print(f"  Overall risk : {s['overall_risk']}")
        print(f"  Critical IDs : {s['critical_threat_ids'] or 'None'}")
        print("=" * 72 + "\n")


def main() -> None:
    """Smoke test: generate and print the threat model."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler("logs/task22_security.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    try:
        registry = ThreatRegistry()
        scorer   = ThreatScorer(registry)
        report   = scorer.score()
        scorer.print_risk_matrix(report)
        print("[OK] Threat model written to logs/threat_model.json")
    except Exception as e:
        logger.critical(f"Threat model generation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
