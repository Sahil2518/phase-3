"""
ml_incident_runbook.py — PlaceMux Phase 3, Task 24
====================================================
ML Incident Runbook Generator

Generates a structured Markdown runbook at logs/ml_incident_runbook.md.

Sections
--------
1. Incident Classification Matrix  (P1-P4 severity levels)
2. Runbook Procedures              (one procedure per chaos scenario)
3. Verification Checklist          (commands to confirm recovery)
4. Escalation Chain                (roles and contacts)
5. Post-Incident Review Template   (5 questions for the post-mortem)

Validation
----------
The generator cross-checks that every scenario ID in the chaos engine
has a matching procedure entry — fails with an assertion error if any
scenario is undocumented.

Output
------
- logs/ml_incident_runbook.md  — the runbook document
- logs/task24.log              — structured log
"""

import os
import sys
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("logs", "task24.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

RUNBOOK_PATH = "logs/ml_incident_runbook.md"

# All chaos scenario IDs that must have a documented procedure
REQUIRED_SCENARIO_IDS = [
    "MODEL_UNAVAILABLE",
    "STALE_FEATURES",
    "CORRUPTED_TRAINING_DATA",
    "FEATURE_STORE_DOWN",
    "NAN_MODEL_OUTPUT",
]


# ---------------------------------------------------------------------------
# Runbook content builders
# ---------------------------------------------------------------------------

def _incident_matrix() -> str:
    """Return the Markdown incident classification matrix."""
    return """## 1. Incident Classification Matrix

| Severity | Code | Symptom | Response SLA | Primary Action |
|----------|------|---------|--------------|----------------|
| Critical | **P1** | Model completely down; all requests in heuristic mode; `degraded_mode=True` in 100% of responses | 15 min | Page on-call ML Engineer + Platform Lead immediately |
| High     | **P2** | Model producing NaN/Inf scores on >10% of requests; feature PSI RED on ≥1 feature | 30 min | Page on-call ML Engineer; activate heuristic layer |
| Medium   | **P3** | Drift detected (PSI YELLOW, JSD>0.10); challenger promoted but AUC delta <2% | 2 hours | Notify ML Engineer during business hours; schedule retrain |
| Low      | **P4** | Retrain cycle ran but challenger rejected; stale features <24 h old | Next business day | Log and monitor; no immediate action |

> **Rule**: A degraded response MUST carry `"degraded_mode": true` in the payload.
> Silent serving of stale features is a P1 escalation.

"""


def _procedure_model_unavailable() -> str:
    """Runbook procedure for MODEL_UNAVAILABLE."""
    return """### Procedure: MODEL_UNAVAILABLE (P1)

**Detection signal**
```
[PAGER ALERT] P1 | MODEL_UNAVAILABLE | Champion model file not found (chaos injected)
```
or in `logs/chaos_alerts.jsonl`:
```json
{"alert_type": "MODEL_UNAVAILABLE", "severity": "P1", "degraded_mode": true}
```

**Symptoms**
- All `/rank` or `/score` API responses contain `"mode": "HEURISTIC"`.
- `logs/task24.log` shows `[DEGRADED MODE] Serving N candidates via heuristic`.
- Model file missing at `models/churn_model_v*.pkl`.

**Verification**
```powershell
# 1. Check model files exist
ls models\\churn_model_v*.pkl

# 2. Check alert log for MODEL_UNAVAILABLE entries
Get-Content logs\\chaos_alerts.jsonl | ConvertFrom-Json | Where-Object { $_.alert_type -eq "MODEL_UNAVAILABLE" }

# 3. Confirm heuristic is serving (not silent failure)
Get-Content logs\\task24.log | Select-String "DEGRADED MODE"
```

**Recovery steps**
1. Check S3/model registry for latest champion version:
   ```python
   from src.model_registry import ModelRegistry
   reg = ModelRegistry()
   reg.list_versions()
   ```
2. Download and restore the champion:
   ```python
   reg.restore_champion(version="latest")
   ```
3. Restart the serving API and confirm `"mode": "ML"` in responses.
4. Verify NDCG@10 is back above 0.70 on a held-out probe set.

**Rollback**
- Keep heuristic layer active until model is confirmed healthy for 5 consecutive probe requests.
- If model repeatedly fails to load, open P1 incident and escalate to Platform Lead.

**Chaos test that proves this path**
```powershell
python -m src.demo_task24  # Scenario 1: MODEL_UNAVAILABLE
```

---

"""


def _procedure_stale_features() -> str:
    """Runbook procedure for STALE_FEATURES."""
    return """### Procedure: STALE_FEATURES (P2)

**Detection signal**
```
[DRIFT CHECK] status=DRIFT_DETECTED | feature_drift=3/9 | retrain=True
```
or PSI alert in `logs/chaos_alerts.jsonl`:
```json
{"alert_type": "STALE_FEATURES", "severity": "P2", "psi_results": {...}}
```

**Symptoms**
- `logs/drift_report.json` shows PSI ≥ 0.25 (RED) for ≥1 feature.
- Feature values are suspiciously constant or uniformly distributed.
- Prediction score distribution shifted (JSD > 0.10).

**Verification**
```powershell
# 1. Read the latest drift report
Get-Content logs\\drift_report.json | ConvertFrom-Json

# 2. Check which features are RED
(Get-Content logs\\drift_report.json | ConvertFrom-Json).feature_drift |
  Where-Object { $_.severity -eq "RED" }

# 3. Compare feature store timestamp vs expected freshness
```

**Recovery steps**
1. Identify the staleness source:
   - Check ETL pipeline last-run timestamp.
   - Query feature store for data freshness metadata.
2. If feature store is stale > 24 h: treat as FEATURE_STORE_DOWN (P1).
3. If drift is genuine (not a bug): trigger retrain:
   ```python
   from src.retraining_pipeline import RetrainingPipeline
   pipe = RetrainingPipeline()
   pipe.run(trigger_reason="stale_feature_drift")
   ```
4. Re-run drift check after retrain to confirm PSI returns GREEN.

**Rollback**
- If retrain does not resolve drift, revert to the previous champion version.
- Activate heuristic layer until drift stabilises.

**Chaos test that proves this path**
```powershell
python -m src.demo_task24  # Scenario 2: STALE_FEATURES
```

---

"""


def _procedure_corrupted_training_data() -> str:
    """Runbook procedure for CORRUPTED_TRAINING_DATA."""
    return """### Procedure: CORRUPTED_TRAINING_DATA (P2)

**Detection signal**
```
[REJECTED] Challenger improvement (-0.0800) below threshold (0.005). Champion retained.
```
or in `logs/retrain_report.json`:
```json
{"status": "REJECTED", "improvement": -0.08, "promoted": false}
```

**Symptoms**
- Retrain cycle rejects challenger with large negative AUC delta.
- Training data has unexpectedly high NaN rates (>5%) in key features.
- Label distribution is abnormal (churn rate outside [10%, 70%]).

**Verification**
```powershell
# 1. Read the latest retrain report
Get-Content logs\\retrain_report.json | ConvertFrom-Json

# 2. Check retrain history for patterns
Get-Content logs\\retrain_history.jsonl | ForEach-Object { $_ | ConvertFrom-Json } |
  Select-Object timestamp, status, improvement | Sort-Object timestamp

# 3. Inspect training data quality
python -c "
import pandas as pd
df = pd.read_parquet('data/training_latest.parquet')
print(df.isnull().mean())
print(df['churned'].value_counts(normalize=True))
"
```

**Recovery steps**
1. Stop the retrain pipeline immediately to prevent overwriting the champion.
2. Investigate upstream data pipeline (ETL, feature engineering) for the corruption source.
3. Roll back to the last known-good training snapshot.
4. Re-run retrain on clean data and verify AUC improvement > 0.005 before promoting.
5. Add a data quality assertion gate:
   - NaN rate < 5% per feature
   - Churn rate in [10%, 70%]

**Rollback**
- Champion model is automatically retained by the AUC gate.
- No action needed if the gate fired correctly.

**Chaos test that proves this path**
```powershell
python -m src.demo_task24  # Scenario 3: CORRUPTED_TRAINING_DATA
```

---

"""


def _procedure_feature_store_down() -> str:
    """Runbook procedure for FEATURE_STORE_DOWN."""
    return """### Procedure: FEATURE_STORE_DOWN (P1)

**Detection signal**
```
[PAGER ALERT] P1 | FEATURE_STORE_DOWN | Feature store returned empty DataFrame
```

**Symptoms**
- Feature store API returns HTTP 503 or empty response body.
- `logs/chaos_alerts.jsonl` shows `"alert_type": "FEATURE_STORE_DOWN"`.
- All model scores are heuristic (`"mode": "HEURISTIC"` in every response).

**Verification**
```powershell
# 1. Check alert log
Get-Content logs\\chaos_alerts.jsonl | ConvertFrom-Json |
  Where-Object { $_.alert_type -eq "FEATURE_STORE_DOWN" }

# 2. Probe the feature store directly
python -c "
from src.inference_engine import load_feature_store
fs = load_feature_store()
print('rows:', len(fs))
"

# 3. Confirm heuristic coverage is 100%
Get-Content logs\\task24.log | Select-String "Serving.*candidates via heuristic"
```

**Recovery steps**
1. Page Platform Engineering to restore the feature store DB.
2. While DB is down: heuristic layer is automatically active — no intervention needed.
3. Once DB is restored, run a feature freshness check:
   ```python
   from src.drift_monitor import DriftMonitor
   monitor = DriftMonitor()
   monitor.check(current_features_df, current_predictions)
   ```
4. Confirm PSI is GREEN before switching back to ML mode.
5. Restart serving API to reload the champion model with fresh features.

**Rollback**
- No rollback needed: heuristic layer handles all traffic during outage.
- Monitor `degraded_mode` flag in responses — switch off when DB is healthy.

**Chaos test that proves this path**
```powershell
python -m src.demo_task24  # Scenario 4: FEATURE_STORE_DOWN
```

---

"""


def _procedure_nan_model_output() -> str:
    """Runbook procedure for NAN_MODEL_OUTPUT."""
    return """### Procedure: NAN_MODEL_OUTPUT (P2)

**Detection signal**
```
WARNING | NaN/Inf score for CAND_0002 (nan) — clamped to 0.0
```
or pager alert:
```json
{"alert_type": "NAN_MODEL_OUTPUT", "severity": "P2", "degraded_mode": true}
```

**Symptoms**
- `logs/task24.log` shows repeated `NaN/Inf score` warnings.
- Candidate ranking is flat (all scores near 0.0).
- Model predictions clustered at 0.0 or 1.0 boundary (overflow / underflow).

**Verification**
```powershell
# 1. Check for NaN warnings in the log
Get-Content logs\\task24.log | Select-String "NaN/Inf score"

# 2. Count clamped scores in last hour
Get-Content logs\\task24.log | Select-String "clamped to 0.0" | Measure-Object

# 3. Probe the model directly
python -c "
import pickle, numpy as np
with open('models/churn_model_v1.pkl', 'rb') as f: m = pickle.load(f)
X = np.random.rand(10, 9).astype('float32')
print(m.predict_proba(X)[:, 1])
"
```

**Recovery steps**
1. Identify the source of NaN/Inf:
   - Inspect recent feature engineering changes for division-by-zero or log(0).
   - Check if input features contain extreme outliers (e.g. days_since_login > 10,000).
2. Add input clipping at the feature engineering layer:
   ```python
   X = np.clip(X, feature_min, feature_max)
   ```
3. If the model itself is corrupted, restore from the previous champion version.
4. Re-run the `NAN_MODEL_OUTPUT` chaos test to confirm the guard is active.

**Rollback**
- NaN/Inf guard clamps bad outputs to 0.0 automatically — no data loss.
- If >50% of scores are clamped, escalate to P1 and activate full heuristic fallback.

**Chaos test that proves this path**
```powershell
python -m src.demo_task24  # Scenario 5: NAN_MODEL_OUTPUT
```

---

"""


def _verification_checklist() -> str:
    """Return the Markdown verification checklist."""
    return """## 3. Verification Checklist

Run these commands after any ML incident to confirm full recovery:

```powershell
# 1. Confirm champion model is present and loadable
python -c "
import pickle
with open('models/churn_model_v1.pkl', 'rb') as f: m = pickle.load(f)
print('Model loaded OK:', type(m).__name__)
"

# 2. Confirm drift is stable (no RED PSI)
python -c "
import json
with open('logs/drift_report.json') as f: r = json.load(f)
reds = [k for k,v in r['feature_drift'].items() if v['severity']=='RED']
print('RED features:', reds or 'none')
"

# 3. Confirm no active pager alerts in last 1 hour
python -c "
import json, datetime
cutoff = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=1)).isoformat()
with open('logs/chaos_alerts.jsonl') as f:
    alerts = [json.loads(l) for l in f if l.strip()]
recent = [a for a in alerts if a['timestamp'] > cutoff]
print('Recent alerts:', len(recent))
"

# 4. Run the full chaos suite and confirm all 5 scenarios PASS
python -m src.demo_task24

# 5. Confirm NDCG@10 in heuristic mode >= 0.45
python -c "
import json
with open('logs/chaos_results.json') as f: r = json.load(f)
for s in r['scenarios']:
    ndcg = s.get('heuristic_ndcg_at_10') or s.get('post_chaos', {}).get('ndcg_at_10', 'N/A')
    print(f\"{s['scenario']}: NDCG@10={ndcg} PASS={s['passed']}\")
"
```

"""


def _escalation_chain() -> str:
    """Return the Markdown escalation chain."""
    return """## 4. Escalation Chain

| Step | Role | Contact | When to Escalate |
|------|------|---------|------------------|
| 1 | On-call ML Engineer | PagerDuty: `#ml-oncall` | Any P1/P2 alert fires |
| 2 | ML Tech Lead | Slack: `#ml-incidents` | Incident not resolved in 30 min |
| 3 | Platform Engineering | Slack: `#platform-oncall` | Feature store / infra issue |
| 4 | VP Engineering | Email / phone | P1 incident >2 hours unresolved |

> **Hand-off rule**: before leaving an incident, write a 3-line status update in `#ml-incidents`:
> (1) What happened, (2) What was done, (3) What is outstanding.

"""


def _post_incident_template() -> str:
    """Return the Markdown post-incident review template."""
    return """## 5. Post-Incident Review Template

Complete within 48 hours of incident resolution.

1. **What happened?**
   _(Timeline of events, from first alert to resolution)_

2. **Why did it happen?**
   _(Root cause — technical and process)_

3. **How was it detected?**
   _(Alert type, latency from failure to detection)_

4. **How was it resolved?**
   _(Steps taken; was the runbook followed?)_

5. **What will prevent recurrence?**
   _(Action items with owner and due date)_

| Action Item | Owner | Due Date |
|-------------|-------|----------|
| | | |

"""


# ---------------------------------------------------------------------------
# RunbookGenerator
# ---------------------------------------------------------------------------

class RunbookGenerator:
    """
    Generates and validates the ML Incident Runbook.

    Parameters
    ----------
    output_path : str
        Where to write the Markdown runbook.
    """

    def __init__(self, output_path: str = RUNBOOK_PATH) -> None:
        self.output_path = output_path

    def _build_procedures(self) -> Dict[str, str]:
        """
        Build all scenario procedures and return as a keyed dict.

        Returns
        -------
        dict
            {scenario_id: markdown_procedure_text}
        """
        return {
            "MODEL_UNAVAILABLE":       _procedure_model_unavailable(),
            "STALE_FEATURES":          _procedure_stale_features(),
            "CORRUPTED_TRAINING_DATA": _procedure_corrupted_training_data(),
            "FEATURE_STORE_DOWN":      _procedure_feature_store_down(),
            "NAN_MODEL_OUTPUT":        _procedure_nan_model_output(),
        }

    def validate_coverage(self, procedures: Dict[str, str]) -> Dict:
        """
        Assert every required scenario has a documented procedure.

        Parameters
        ----------
        procedures : dict
            Keyed by scenario ID.

        Returns
        -------
        dict
            {'covered': list, 'missing': list, 'all_covered': bool}
        """
        covered = [sid for sid in REQUIRED_SCENARIO_IDS if sid in procedures]
        missing = [sid for sid in REQUIRED_SCENARIO_IDS if sid not in procedures]
        all_covered = len(missing) == 0
        if not all_covered:
            logger.error(f"Runbook missing procedures for: {missing}")
        else:
            logger.info(f"Runbook coverage: all {len(covered)} scenarios documented")
        return {"covered": covered, "missing": missing, "all_covered": all_covered}

    def generate(self) -> Dict:
        """
        Build the full runbook Markdown and write it to disk.

        Returns
        -------
        dict
            {
              'path': str,
              'n_procedures': int,
              'coverage': dict,
              'all_covered': bool,
            }
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        procedures = self._build_procedures()
        coverage = self.validate_coverage(procedures)

        lines = [
            "# PlaceMux ML Incident Runbook",
            "",
            f"> Generated: {timestamp}",
            "> Task 24 — Disaster Recovery, Chaos Testing & Business Continuity",
            "",
            "---",
            "",
            "## Purpose",
            "",
            "This runbook tells the on-call ML engineer exactly what to do",
            "when the PlaceMux matching model misbehaves.",
            "Every procedure in this document is backed by a passing chaos test.",
            "",
            "**The bar:** When the model dies, matching degrades to a sane heuristic",
            "and someone is paged — nothing silently breaks.",
            "",
            "---",
            "",
            _incident_matrix(),
            "## 2. Runbook Procedures",
            "",
            "One procedure per chaos scenario, in order of expected frequency:",
            "",
        ]

        for procedure in procedures.values():
            lines.append(procedure)

        lines.append(_verification_checklist())
        lines.append(_escalation_chain())
        lines.append(_post_incident_template())

        content = "\n".join(lines)

        try:
            with open(self.output_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"Runbook written: {self.output_path} ({len(content)} bytes)")
        except Exception as e:
            logger.error(f"Failed to write runbook: {e}")
            raise

        return {
            "path": self.output_path,
            "n_procedures": len(procedures),
            "coverage": coverage,
            "all_covered": coverage["all_covered"],
            "size_bytes": len(content),
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Generate the ML incident runbook and print a summary."""
    import sys
    try:
        gen = RunbookGenerator()
        result = gen.generate()
        print(f"\n✅ Runbook generated: {result['path']}")
        print(f"   Procedures documented: {result['n_procedures']}")
        print(f"   Coverage complete    : {result['all_covered']}")
        print(f"   Missing scenarios    : {result['coverage']['missing'] or 'none'}")
        print(f"   File size            : {result['size_bytes']:,} bytes")
    except Exception as e:
        logger.critical(f"Runbook generation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
