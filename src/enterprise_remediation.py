"""
enterprise_remediation.py -- PlaceMux Phase 3, Task 20 (Stage D)
=================================================================
Reads pilot metrics, fairness, and latency reports then produces a
prioritised remediation list before the real enterprise pilot.

Severity classification:
  CRITICAL  — metric below hard floor (blocks go-live)
  HIGH      — metric within 10% of threshold
  MEDIUM    — informational / best-practice improvements

Output: logs/task20_remediation_list.json
        Each item: { issue, severity, owner_role, action, metric_gap }
"""

import os
import sys
import json
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hard floors (CRITICAL if breached)
# ---------------------------------------------------------------------------
FLOOR_PRECISION_AT_K = 0.45
FLOOR_PARITY_GAP = 0.25      # gap above this = CRITICAL
FLOOR_P95_MS = 200.0
FLOOR_P50_MS = 80.0

# ---------------------------------------------------------------------------
# Acceptance bars (HIGH if within 10% of bar)
# ---------------------------------------------------------------------------
BAR_PRECISION_AT_K = 0.60
BAR_PARITY_GAP = 0.15
BAR_P95_MS = 100.0
BAR_P50_MS = 30.0


class RemediationItem:
    """
    Single remediation finding.

    Parameters
    ----------
    issue : str
        Short description of the problem.
    severity : str
        CRITICAL | HIGH | MEDIUM
    owner_role : str
        Role responsible for resolving (e.g. 'ML Engineer').
    action : str
        Concrete next step.
    metric_gap : Optional[float]
        Numeric gap from acceptance bar (positive = worse than bar).
    """

    def __init__(self, issue: str, severity: str, owner_role: str,
                 action: str, metric_gap: Optional[float] = None):
        self.issue = issue
        self.severity = severity
        self.owner_role = owner_role
        self.action = action
        self.metric_gap = metric_gap

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to dict for JSON output."""
        return {
            "issue": self.issue,
            "severity": self.severity,
            "owner_role": self.owner_role,
            "action": self.action,
            "metric_gap": self.metric_gap,
        }


class EnterpriseRemediationGenerator:
    """
    Reads pilot and evaluation reports and generates a prioritised
    remediation list with severity labels.

    Parameters
    ----------
    pilot_metrics_path : str
        Path to task20_pilot_metrics.json.
    fairness_report_path : str
        Path to task20_fairness_report.json.
    latency_report_path : str
        Path to task20_latency_report.json.
    """

    def __init__(self,
                 pilot_metrics_path: str = "logs/task20_pilot_metrics.json",
                 fairness_report_path: str = "logs/task20_fairness_report.json",
                 latency_report_path: str = "logs/task20_latency_report.json"):
        self.pilot_metrics_path = pilot_metrics_path
        self.fairness_report_path = fairness_report_path
        self.latency_report_path = latency_report_path
        self.items: List[RemediationItem] = []

    def _load_json(self, path: str) -> Dict[str, Any]:
        """
        Load a JSON report safely.

        Parameters
        ----------
        path : str

        Returns
        -------
        dict
            Parsed JSON, or empty dict on error.
        """
        if not os.path.exists(path):
            logger.error(f"Report not found: {path}")
            return {}
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            return {}

    def _classify(self, value: float, bar: float, floor: float,
                  higher_is_better: bool = True) -> str:
        """
        Return CRITICAL / HIGH / MEDIUM severity based on value vs bar.

        Parameters
        ----------
        value : float
        bar : float
        floor : float
        higher_is_better : bool
            True for precision/recall (larger = better), False for latency/gap.

        Returns
        -------
        str
        """
        if higher_is_better:
            if value < floor:
                return "CRITICAL"
            if value < bar * 1.10:
                return "HIGH"
        else:
            if value > floor:
                return "CRITICAL"
            if value > bar * 0.90:
                return "HIGH"
        return "MEDIUM"

    # ------------------------------------------------------------------
    # Rule-based checks
    # ------------------------------------------------------------------
    def _check_quality(self, pilot: Dict[str, Any]) -> None:
        """
        Generate quality-related remediation items.

        Parameters
        ----------
        pilot : dict
            Parsed pilot_metrics.json.
        """
        ips = pilot.get("ips_ranker", {})
        p_at_k = ips.get("precision_at_10", ips.get("precision_at_5", None))
        if p_at_k is None:
            self.items.append(RemediationItem(
                issue="Pilot metrics JSON missing — cannot evaluate quality.",
                severity="CRITICAL",
                owner_role="ML Engineer",
                action="Re-run enterprise_pilot_runner.py and confirm JSON output.",
                metric_gap=None,
            ))
            return

        gap = round(BAR_PRECISION_AT_K - p_at_k, 4)
        sev = self._classify(p_at_k, BAR_PRECISION_AT_K, FLOOR_PRECISION_AT_K, higher_is_better=True)
        if sev != "MEDIUM" or gap > 0:
            self.items.append(RemediationItem(
                issue=f"Precision@10 ({p_at_k:.4f}) is below acceptance bar ({BAR_PRECISION_AT_K}).",
                severity=sev,
                owner_role="ML Engineer",
                action=(
                    "Upgrade from SimpleIPSRanker to LightGBM LambdaMART (ltr_model.py). "
                    "Increase training data volume. Tune IPS weight cap."
                ),
                metric_gap=gap,
            ))

        ndcg = ips.get("ndcg_at_10", ips.get("ndcg_at_5", None))
        if ndcg is not None and ndcg < 0.50:
            self.items.append(RemediationItem(
                issue=f"NDCG@10 ({ndcg:.4f}) is below 0.50 — ranking order weak.",
                severity="HIGH",
                owner_role="ML Engineer",
                action="Add more discriminative features: recency, apply-rate signals, LLM embedding similarity.",
                metric_gap=round(0.50 - ndcg, 4),
            ))

    def _check_fairness(self, fairness: Dict[str, Any]) -> None:
        """
        Generate fairness-related remediation items.

        Parameters
        ----------
        fairness : dict
        """
        fair_data = fairness.get("fairness", fairness)
        parity_gap = fair_data.get("parity_gap", None)

        if parity_gap is None:
            self.items.append(RemediationItem(
                issue="Fairness report missing parity_gap — evaluation was skipped.",
                severity="HIGH",
                owner_role="ML Engineer / Legal",
                action="Ensure demographic attributes are present in candidate data and re-run evaluator.",
                metric_gap=None,
            ))
            return

        sev = self._classify(parity_gap, BAR_PARITY_GAP, FLOOR_PARITY_GAP, higher_is_better=False)
        gap = round(parity_gap - BAR_PARITY_GAP, 4)
        if sev in ("CRITICAL", "HIGH") or parity_gap > BAR_PARITY_GAP:
            self.items.append(RemediationItem(
                issue=f"Demographic parity gap ({parity_gap:.4f}) exceeds bar ({BAR_PARITY_GAP}).",
                severity=sev,
                owner_role="ML Engineer / Ethics Lead",
                action=(
                    "Implement re-ranking with fairness constraints (e.g. max-min fairness). "
                    "Audit training data for representation imbalance. "
                    "Add group recall monitoring to SLO dashboard."
                ),
                metric_gap=gap,
            ))

        # Check individual groups for very low recall
        group_recalls = fair_data.get("group_recalls", {})
        max_recall = fair_data.get("max_group_recall", 1.0)
        for group, recall in group_recalls.items():
            if max_recall > 0 and recall < 0.5 * max_recall:
                self.items.append(RemediationItem(
                    issue=f"Group '{group}' recall ({recall:.4f}) is < 50% of best group ({max_recall:.4f}).",
                    severity="CRITICAL",
                    owner_role="ML Engineer / Ethics Lead",
                    action=f"Investigate training data representation for {group}. Apply over-sampling or fairness-aware loss.",
                    metric_gap=round(max_recall - recall, 4),
                ))

    def _check_latency(self, latency_report: Dict[str, Any]) -> None:
        """
        Generate latency-related remediation items.

        Parameters
        ----------
        latency_report : dict
        """
        lat = latency_report.get("latency", latency_report)
        p50 = lat.get("p50_ms", None)
        p95 = lat.get("p95_ms", None)

        if p50 is None or p95 is None:
            self.items.append(RemediationItem(
                issue="Latency report missing — benchmark did not run.",
                severity="HIGH",
                owner_role="ML Engineer / Platform",
                action="Re-run enterprise_fairness_evaluator.py latency benchmark.",
                metric_gap=None,
            ))
            return

        if p50 > BAR_P50_MS:
            sev = "CRITICAL" if p50 > FLOOR_P50_MS else "HIGH"
            self.items.append(RemediationItem(
                issue=f"p50 latency ({p50:.1f}ms) exceeds bar ({BAR_P50_MS}ms).",
                severity=sev,
                owner_role="ML Engineer / Platform",
                action=(
                    "Pre-compute job feature vectors at ingestion time. "
                    "Cache compiled tenant matrices. "
                    "Batch candidate scoring using numpy vectorisation."
                ),
                metric_gap=round(p50 - BAR_P50_MS, 2),
            ))

        if p95 > BAR_P95_MS:
            sev = "CRITICAL" if p95 > FLOOR_P95_MS else "HIGH"
            self.items.append(RemediationItem(
                issue=f"p95 latency ({p95:.1f}ms) exceeds bar ({BAR_P95_MS}ms).",
                severity=sev,
                owner_role="Platform / ML Engineer",
                action=(
                    "Deploy inference behind async worker pool. "
                    "Add p95 latency SLO to monitoring dashboard. "
                    "Investigate tail latency causes (GC pauses, cold-start)."
                ),
                metric_gap=round(p95 - BAR_P95_MS, 2),
            ))

    def _add_standing_items(self) -> None:
        """
        Add standing pre-pilot checklist items regardless of metrics.

        These represent best-practice requirements that must be confirmed
        before any real enterprise pilot, irrespective of offline results.
        """
        self.items.append(RemediationItem(
            issue="No agreed online acceptance criteria with AcmeCorp stakeholders.",
            severity="CRITICAL",
            owner_role="Product / BD",
            action=(
                "Schedule acceptance criteria review session with AcmeCorp hiring managers. "
                "Define: precision bar, response time SLA, fairness commitment in contract."
            ),
            metric_gap=None,
        ))
        self.items.append(RemediationItem(
            issue="Offline P@10 gap vs online CTR not validated.",
            severity="HIGH",
            owner_role="ML Engineer / Data Scientist",
            action=(
                "Run a shadow deployment: log IPS-ranked results alongside heuristic results "
                "and measure CTR difference before full rollout."
            ),
            metric_gap=None,
        ))
        self.items.append(RemediationItem(
            issue="Model card not reviewed by AcmeCorp compliance team.",
            severity="HIGH",
            owner_role="ML Engineer / Legal",
            action=(
                "Share model_card_churn_model_v*.md with AcmeCorp DPO. "
                "Add AcmeCorp-specific data lineage section to model card."
            ),
            metric_gap=None,
        ))
        self.items.append(RemediationItem(
            issue="Drift monitoring not enabled for AcmeCorp tenant slice.",
            severity="MEDIUM",
            owner_role="ML Engineer",
            action=(
                "Add per-tenant PSI / JSD monitoring (drift_monitor.py) scoped to acmecorp. "
                "Set alert threshold at PSI > 0.1."
            ),
            metric_gap=None,
        ))
        self.items.append(RemediationItem(
            issue="Model rollback procedure not tested for AcmeCorp.",
            severity="MEDIUM",
            owner_role="ML Engineer / Platform",
            action=(
                "Register AcmeCorp pilot model version in model_registry.py. "
                "Test champion/challenger swap using retraining_pipeline.py."
            ),
            metric_gap=None,
        ))

    # ------------------------------------------------------------------
    # Main generate
    # ------------------------------------------------------------------
    def generate(self) -> List[Dict[str, Any]]:
        """
        Load all reports and generate the prioritised remediation list.

        Returns
        -------
        List[dict]
            Sorted remediation items (CRITICAL first, then HIGH, MEDIUM).
        """
        logger.info("=== Stage D: Generating Remediation List ===")

        pilot = self._load_json(self.pilot_metrics_path)
        fairness = self._load_json(self.fairness_report_path)
        latency = self._load_json(self.latency_report_path)

        self._check_quality(pilot)
        self._check_fairness(fairness)
        self._check_latency(latency)
        self._add_standing_items()

        # Sort: CRITICAL -> HIGH -> MEDIUM
        severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2}
        self.items.sort(key=lambda x: severity_order.get(x.severity, 3))

        logger.info(f"Remediation list: {len(self.items)} items "
                    f"({sum(1 for i in self.items if i.severity=='CRITICAL')} CRITICAL, "
                    f"{sum(1 for i in self.items if i.severity=='HIGH')} HIGH, "
                    f"{sum(1 for i in self.items if i.severity=='MEDIUM')} MEDIUM)")

        return [i.to_dict() for i in self.items]

    def save(self, out_path: str = "logs/task20_remediation_list.json") -> None:
        """
        Persist the remediation list to JSON.

        Parameters
        ----------
        out_path : str
        """
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        items_dicts = [i.to_dict() for i in self.items]
        payload = {
            "tenant_id": "acmecorp",
            "total_items": len(items_dicts),
            "critical_count": sum(1 for i in self.items if i.severity == "CRITICAL"),
            "high_count": sum(1 for i in self.items if i.severity == "HIGH"),
            "medium_count": sum(1 for i in self.items if i.severity == "MEDIUM"),
            "items": items_dicts,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Remediation list saved to {out_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    gen = EnterpriseRemediationGenerator()
    items = gen.generate()
    gen.save()
    print(json.dumps(items, indent=2))
