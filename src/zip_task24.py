"""
zip_task24.py — PlaceMux Phase 3, Task 24
Packages the Task 24 deliverables into a versioned ZIP archive.
"""
import zipfile
import os
import datetime

task_num = 24
date_str = datetime.date.today().strftime("%Y%m%d")
zip_name = f"placemux_task{task_num:02d}_{date_str}.zip"

include_src = [
    "chaos_engine.py",
    "graceful_degradation.py",
    "ml_incident_runbook.py",
    "demo_task24.py",
    "zip_task24.py",
    # upstream dependencies used by the chaos engine
    "drift_monitor.py",
    "retraining_pipeline.py",
]

include_logs = [
    "chaos_results.json",
    "chaos_alerts.jsonl",
    "ml_incident_runbook.md",
    "task24.log",
]

with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
    for fname in include_src:
        path = os.path.join("src", fname)
        if os.path.exists(path):
            zf.write(path)
        else:
            print(f"[WARN] src file not found: {path}")

    for fname in include_logs:
        path = os.path.join("logs", fname)
        if os.path.exists(path):
            zf.write(path)
        else:
            print(f"[WARN] log file not found: {path}")

    for extra in ["run_task24.bat", "requirements.txt", "README.md"]:
        if os.path.exists(extra):
            zf.write(extra)

print(f"[OK] ZIP created: {zip_name}")
