"""
zip_task22.py — PlaceMux Phase 3, Task 22
=========================================
Packages all Task 22 deliverables into a versioned ZIP archive.
(Rule 1: Always create a ZIP at the end of every full task.)
"""

import zipfile
import os
import datetime
import sys

TASK_NUM = 22
DATE_STR = datetime.date.today().strftime("%Y%m%d")
ZIP_NAME = f"placemux_task{TASK_NUM:02d}_{DATE_STR}.zip"

INCLUDE_DIRS  = ["src", "models", "logs"]
INCLUDE_FILES = [
    f"run_task{TASK_NUM:02d}.bat",
    "requirements.txt",
    "README.md",
]

# Task-specific files to always include
TASK_SPECIFIC = [
    "src/drift_monitor.py",
    "src/retraining_pipeline.py",
    "src/demo_task22.py",
    "src/zip_task22.py",
    "logs/drift_report.json",
    "logs/drift_history.jsonl",
    "logs/retrain_report.json",
    "logs/retrain_history.jsonl",
    "logs/task22.log",
]


def create_zip() -> None:
    """Create the ZIP archive for Task 22 deliverables."""
    with zipfile.ZipFile(ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as zf:
        # Walk standard directories
        for folder in INCLUDE_DIRS:
            if not os.path.isdir(folder):
                continue
            for root, dirs, files in os.walk(folder):
                # Exclude __pycache__ and venv
                dirs[:] = [d for d in dirs if d not in ("__pycache__", "venv", ".git")]
                for file in files:
                    full_path = os.path.join(root, file)
                    zf.write(full_path)

        # Root-level files
        for f in INCLUDE_FILES:
            if os.path.exists(f):
                zf.write(f)

    print(f"[OK] ZIP created: {ZIP_NAME}")


if __name__ == "__main__":
    try:
        create_zip()
    except Exception as e:
        print(f"[ERROR] ZIP creation failed: {e}", file=sys.stderr)
        sys.exit(1)
