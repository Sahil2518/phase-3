"""
zip_task20.py -- PlaceMux Phase 3, Task 20
==========================================
Packages Task 20 deliverables into a versioned ZIP archive.
"""
import zipfile
import os
import datetime
import sys


def create_zip():
    """Create the Task 20 ZIP archive including all deliverables."""
    task_num = 20
    date_str = datetime.date.today().strftime("%Y%m%d")
    zip_name = f"placemux_task{task_num:02d}_{date_str}.zip"

    task20_src_files = [
        "enterprise_pilot_dataset.py",
        "enterprise_pilot_runner.py",
        "enterprise_fairness_evaluator.py",
        "enterprise_remediation.py",
        "demo_task20.py",
        "zip_task20.py",
    ]

    task20_log_files = [
        "task20_pilot_metrics.json",
        "task20_fairness_report.json",
        "task20_latency_report.json",
        "task20_remediation_list.json",
        "task20.log",
        "task20_candidates.csv",
        "task20_jobs.csv",
        "task20_interactions.csv",
        "task20_train.csv",
        "task20_test.csv",
    ]

    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
        # Task 20 specific source files
        for fname in task20_src_files:
            fpath = os.path.join("src", fname)
            if os.path.exists(fpath):
                zf.write(fpath)
                print(f"  Added: {fpath}")

        # Task 20 log / output files
        for fname in task20_log_files:
            fpath = os.path.join("logs", fname)
            if os.path.exists(fpath):
                zf.write(fpath)
                print(f"  Added: {fpath}")

        # Batch runner
        bat = "run_task20.bat"
        if os.path.exists(bat):
            zf.write(bat)
            print(f"  Added: {bat}")

        # Project-level files
        for f in ["requirements.txt", "README.md"]:
            if os.path.exists(f):
                zf.write(f)
                print(f"  Added: {f}")

    print(f"\n[OK] ZIP created: {zip_name}")
    return zip_name


if __name__ == "__main__":
    try:
        create_zip()
    except Exception as e:
        print(f"[FAIL] ZIP creation failed: {e}", file=sys.stderr)
        sys.exit(1)
