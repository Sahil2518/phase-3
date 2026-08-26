"""
decision_disclosure.py -- PlaceMux Phase 3, Task 23
====================================================
Automated-Decision Disclosure and Human-Review Path.

Design rationale
----------------
GDPR Art. 22 / DPDP Sec. 12 requires that candidates subject to automated
ranking decisions have the right to:
  1. Know they were subject to an automated decision.
  2. Receive a meaningful explanation of the factors involved.
  3. Request human review of the decision.

This module provides:

DecisionRecord
  A dataclass capturing every material fact about one ranking decision.

DecisionDisclosureEngine
  - log_decision:       Persists every decision to logs/decision_log.jsonl.
  - explain_decision:   Generates a plain-English explanation using SHAP-lite
                        (score delta attribution -- no external libraries).
  - escalate_to_human:  Flags borderline / high-risk decisions for human review.
  - get_disclosure:     Returns the full disclosure bundle for a subject_id.

SHAP-lite attribution
---------------------
For each feature, the contribution is approximated by:
  contribution_i = (feature_i - mean_i) * coefficient_i
where coefficient_i is a pre-stored linear approximation of the feature's
average marginal effect on the model score (derived from a calibration run
on reference data).  This requires no external library and is reproducible.

Metric bar
----------
  100% of decisions logged to JSONL within the same request cycle.
  Borderline cases (score within 0.05 of cutoff) escalated in < 1s.
  Plain-English explanation covers top-3 contributing features.
"""

import os
import sys
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any

os.makedirs("logs", exist_ok=True)
logger = logging.getLogger(__name__)

DECISION_LOG_PATH    = "logs/decision_log.jsonl"
HUMAN_REVIEW_PATH    = "logs/human_review_queue.json"

# ---------------------------------------------------------------------------
# SHAP-lite coefficients (calibrated on reference data, stored statically)
# These represent the average marginal effect of each feature on the score.
# ---------------------------------------------------------------------------

FEATURE_COEFFICIENTS: Dict[str, float] = {
    "sessions_14d":         +0.022,
    "apply_rate_7d":        +0.310,
    "jobs_viewed":          +0.003,
    "profile_completeness": +0.008,
    "days_since_login":     -0.018,
    "is_verified":          +0.040,
    "recruiter_contacts":   +0.025,
    "seniority_match":      +0.120,
}

FEATURE_MEANS: Dict[str, float] = {
    "sessions_14d":         8.5,
    "apply_rate_7d":        0.22,
    "jobs_viewed":          95.0,
    "profile_completeness": 65.0,
    "days_since_login":     28.0,
    "is_verified":          0.6,
    "recruiter_contacts":   5.0,
    "seniority_match":      0.5,
}

FEATURE_DISPLAY_NAMES: Dict[str, str] = {
    "sessions_14d":         "activity in the last 14 days",
    "apply_rate_7d":        "application rate this week",
    "jobs_viewed":          "job-browsing engagement",
    "profile_completeness": "profile completeness",
    "days_since_login":     "days since last login",
    "is_verified":          "profile verification status",
    "recruiter_contacts":   "recruiter interactions",
    "seniority_match":      "seniority match to role",
}


# ---------------------------------------------------------------------------
# DecisionRecord
# ---------------------------------------------------------------------------

@dataclass
class DecisionRecord:
    """
    Captures every material fact about one automated ranking decision.

    Parameters
    ----------
    candidate_id    : str
    job_id          : str
    model_version   : str
    raw_score       : float   Score in [0, 1].
    shortlist_cutoff: float   Minimum score to enter the shortlist.
    rank            : int     Candidate's rank (1 = top).
    total_candidates: int     Total candidates scored for this job.
    outcome         : str     'SHORTLISTED' | 'REJECTED'
    feature_values  : Dict    Feature name -> observed value for this candidate.
    feature_contributions: Dict  Feature name -> attribution score (SHAP-lite).
    timestamp       : str     ISO-8601 UTC.
    borderline      : bool    Score within 0.05 of cutoff.
    escalated       : bool    Whether human review was triggered.
    """

    candidate_id:         str
    job_id:               str
    model_version:        str
    raw_score:            float
    shortlist_cutoff:     float
    rank:                 int
    total_candidates:     int
    outcome:              str
    feature_values:       Dict[str, float] = field(default_factory=dict)
    feature_contributions: Dict[str, float] = field(default_factory=dict)
    timestamp:            str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    borderline:           bool = False
    escalated:            bool = False


# ---------------------------------------------------------------------------
# DecisionDisclosureEngine
# ---------------------------------------------------------------------------

class DecisionDisclosureEngine:
    """
    Records, explains, and escalates automated ranking decisions.

    Parameters
    ----------
    borderline_margin : float  Score margin around the cutoff that triggers
                               human-review escalation (default 0.05).
    low_confidence_threshold : float  Raw score below which confidence is
                               considered insufficient (default 0.55).
    decision_log_path : str
    human_review_path : str
    """

    def __init__(
        self,
        borderline_margin:        float = 0.05,
        low_confidence_threshold: float = 0.55,
        decision_log_path:        str   = DECISION_LOG_PATH,
        human_review_path:        str   = HUMAN_REVIEW_PATH,
    ) -> None:
        """Initialise the disclosure engine."""
        self.borderline_margin        = borderline_margin
        self.low_confidence_threshold = low_confidence_threshold
        self.decision_log_path        = decision_log_path
        self.human_review_path        = human_review_path
        self._review_queue:           List[Dict] = []

        logger.info(
            f"DecisionDisclosureEngine ready: "
            f"borderline_margin={borderline_margin}, "
            f"low_confidence={low_confidence_threshold}"
        )

    # ------------------------------------------------------------------
    # SHAP-lite attribution
    # ------------------------------------------------------------------

    def _compute_attributions(
        self, feature_values: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute SHAP-lite feature attributions using linear approximation.

        contribution_i = (value_i - mean_i) * coefficient_i

        Parameters
        ----------
        feature_values : Dict[str, float]  Observed feature values.

        Returns
        -------
        Dict[str, float]  Feature name -> attribution score.
        """
        attributions = {}
        for feat, value in feature_values.items():
            coef = FEATURE_COEFFICIENTS.get(feat, 0.0)
            mean = FEATURE_MEANS.get(feat, 0.0)
            attributions[feat] = round((value - mean) * coef, 5)
        return attributions

    # ------------------------------------------------------------------
    # Log decision
    # ------------------------------------------------------------------

    def log_decision(self, record: DecisionRecord) -> None:
        """
        Append a DecisionRecord to the JSONL decision log.

        Parameters
        ----------
        record : DecisionRecord
        """
        try:
            with open(self.decision_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(record)) + "\n")
        except Exception as e:
            logger.error(f"Failed to log decision for {record.candidate_id}: {e}")

    # ------------------------------------------------------------------
    # Explain decision
    # ------------------------------------------------------------------

    def explain_decision(self, record: DecisionRecord) -> str:
        """
        Generate a plain-English explanation of a ranking decision.

        Parameters
        ----------
        record : DecisionRecord

        Returns
        -------
        str  Plain-English explanation (DPDP Art. 12 / GDPR Art. 22 compliant).
        """
        # Guard: uninitialized record
        if record is None:
            return "No decision record available. Please contact support."

        contribs = record.feature_contributions
        sorted_contribs = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)
        top3 = sorted_contribs[:3]

        pos_factors = [
            f"{FEATURE_DISPLAY_NAMES.get(f, f)} (+{v:.3f})"
            for f, v in top3 if v > 0
        ]
        neg_factors = [
            f"{FEATURE_DISPLAY_NAMES.get(f, f)} ({v:.3f})"
            for f, v in top3 if v < 0
        ]

        lines = [
            f"Automated Ranking Decision -- {record.candidate_id} for {record.job_id}",
            f"",
            f"Outcome     : {record.outcome}",
            f"Your score  : {record.raw_score:.3f} (cutoff: {record.shortlist_cutoff:.3f})",
            f"Your rank   : #{record.rank} of {record.total_candidates} candidates",
            f"Model used  : {record.model_version}",
            f"Decided at  : {record.timestamp}",
            f"",
            f"Why you received this outcome:",
        ]
        if pos_factors:
            lines.append(f"  POSITIVE factors: {', '.join(pos_factors)}")
        if neg_factors:
            lines.append(f"  NEGATIVE factors: {', '.join(neg_factors)}")
        if not pos_factors and not neg_factors:
            lines.append("  No dominant factors identified.")
        lines += [
            f"",
            f"This decision was made by an automated ranking system. "
            f"You have the right to request human review of this outcome.",
        ]
        if record.borderline:
            lines.append(
                f"Note: Your score was within {self.borderline_margin:.0%} of the "
                f"shortlist cutoff. This case has been flagged for human review."
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Escalate to human review
    # ------------------------------------------------------------------

    def escalate_to_human_review(
        self,
        record: DecisionRecord,
        reason: str = "candidate_request",
    ) -> Dict:
        """
        Create a human-review escalation ticket.

        Escalation is triggered automatically for:
          - Borderline scores (within margin of cutoff)
          - Low model confidence (score < low_confidence_threshold)
          - Explicit candidate request

        Parameters
        ----------
        record : DecisionRecord
        reason : str  Escalation trigger reason.

        Returns
        -------
        dict  Escalation ticket.
        """
        ticket_id = f"HRQ-{record.candidate_id}-{record.job_id}"
        ticket = {
            "ticket_id":       ticket_id,
            "candidate_id":    record.candidate_id,
            "job_id":          record.job_id,
            "raw_score":       record.raw_score,
            "outcome":         record.outcome,
            "rank":            record.rank,
            "model_version":   record.model_version,
            "reason":          reason,
            "escalated_at":    datetime.now(timezone.utc).isoformat(),
            "explanation":     self.explain_decision(record),
            "status":          "PENDING_HUMAN_REVIEW",
        }
        self._review_queue.append(ticket)
        self._save_review_queue()
        logger.warning(
            f"[HumanReview] Ticket {ticket_id} created: "
            f"reason={reason} score={record.raw_score:.3f}"
        )
        return ticket

    def _save_review_queue(self) -> None:
        """Persist the human-review queue to JSON."""
        try:
            with open(self.human_review_path, "w", encoding="utf-8") as f:
                json.dump(self._review_queue, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save human review queue: {e}")

    # ------------------------------------------------------------------
    # Process one decision end-to-end
    # ------------------------------------------------------------------

    def process_decision(
        self,
        candidate_id:     str,
        job_id:           str,
        model_version:    str,
        raw_score:        float,
        shortlist_cutoff: float,
        rank:             int,
        total_candidates: int,
        feature_values:   Dict[str, float],
    ) -> Dict:
        """
        Full pipeline: log -> attribute -> explain -> escalate if needed.

        Parameters
        ----------
        candidate_id     : str
        job_id           : str
        model_version    : str
        raw_score        : float
        shortlist_cutoff : float
        rank             : int
        total_candidates : int
        feature_values   : Dict[str, float]

        Returns
        -------
        dict  {"record": DecisionRecord, "explanation": str, "escalated": bool}
        """
        # Guard: invalid score
        if raw_score is None or not (0.0 <= raw_score <= 1.0):
            logger.warning(
                f"Invalid score {raw_score} for {candidate_id}. "
                f"Defaulting to 0.0."
            )
            raw_score = 0.0

        outcome = "SHORTLISTED" if raw_score >= shortlist_cutoff else "REJECTED"
        borderline = abs(raw_score - shortlist_cutoff) <= self.borderline_margin

        attributions = self._compute_attributions(feature_values)

        record = DecisionRecord(
            candidate_id=candidate_id,
            job_id=job_id,
            model_version=model_version,
            raw_score=raw_score,
            shortlist_cutoff=shortlist_cutoff,
            rank=rank,
            total_candidates=total_candidates,
            outcome=outcome,
            feature_values=feature_values,
            feature_contributions=attributions,
            borderline=borderline,
        )

        self.log_decision(record)
        explanation = self.explain_decision(record)

        escalated = False
        ticket = None
        if borderline:
            ticket = self.escalate_to_human_review(record, reason="borderline_score")
            record.escalated = True
            escalated = True
        elif raw_score < self.low_confidence_threshold:
            ticket = self.escalate_to_human_review(record, reason="low_confidence")
            record.escalated = True
            escalated = True

        return {
            "record":      asdict(record),
            "explanation": explanation,
            "escalated":   escalated,
            "ticket":      ticket,
        }

    # ------------------------------------------------------------------
    # Disclosure bundle for a subject
    # ------------------------------------------------------------------

    def get_disclosure(self, subject_id: str) -> Dict:
        """
        Return the full disclosure bundle for a candidate_id.

        Reads the decision log and returns all records + explanations.

        Parameters
        ----------
        subject_id : str

        Returns
        -------
        dict  {"subject_id", "decisions": [...], "n_decisions": int}
        """
        records = []
        try:
            if os.path.exists(self.decision_log_path):
                with open(self.decision_log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        if rec.get("candidate_id") == subject_id:
                            records.append(rec)
        except Exception as e:
            logger.error(f"Failed to read decision log: {e}")

        return {
            "subject_id":  subject_id,
            "n_decisions": len(records),
            "decisions":   records,
        }

    def queue_length(self) -> int:
        """Return the current length of the human-review queue."""
        return len(self._review_queue)


def main() -> None:
    """Smoke test."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler("logs/task23.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    try:
        engine = DecisionDisclosureEngine()
        result = engine.process_decision(
            candidate_id="CAND_0001",
            job_id="JOB_1234",
            model_version="v2",
            raw_score=0.62,
            shortlist_cutoff=0.60,
            rank=4,
            total_candidates=30,
            feature_values={
                "sessions_14d": 12.0,
                "apply_rate_7d": 0.35,
                "days_since_login": 5.0,
                "profile_completeness": 85.0,
                "is_verified": 1.0,
            },
        )
        print(result["explanation"])
        print(f"Escalated: {result['escalated']}")
        print("[OK] decision_disclosure smoke test passed.")
    except Exception as e:
        logger.critical(f"Smoke test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
