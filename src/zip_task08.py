"""
zip_task08.py -- PlaceMux Phase 3, Task 8
Creates the submission ZIP archive following standing Rule 1.
"""

import os
import sys
import zipfile
import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def create_zip() -> str:
    """
    Package Task 8 deliverables into a versioned ZIP archive.

    Contents
    --------
    src/      -- all Python scripts
    models/   -- trained model artefact
    logs/     -- metrics JSON, PR curve, at-risk list, experiment logs
    run_task08.bat
    requirements.txt
    README.md

    Returns
    -------
    str
        Path to the created ZIP file.
    """
    task_num = 8
    date_str = datetime.date.today().strftime("%Y%m%d")
    zip_name = f"placemux_task{task_num:02d}_{date_str}.zip"

    logger.info(f"Creating ZIP: {zip_name}")

    included = []
    skipped = []

    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
        # Folder contents
        for folder in ["src", "models", "logs"]:
            if not os.path.isdir(folder):
                logger.warning(f"Folder '{folder}' not found -- skipping.")
                continue
            for root, dirs, files in os.walk(folder):
                # Skip __pycache__
                dirs[:] = [d for d in dirs if d != "__pycache__"]
                for file in files:
                    full_path = os.path.join(root, file)
                    zf.write(full_path)
                    included.append(full_path)

        # Top-level files
        for f in [f"run_task{task_num:02d}.bat", "requirements.txt", "README.md"]:
            if os.path.exists(f):
                zf.write(f)
                included.append(f)
            else:
                skipped.append(f)

    logger.info(f"[OK] ZIP created: {zip_name}  ({len(included)} files)")
    if skipped:
        logger.warning(f"Skipped (not found): {skipped}")

    return zip_name


def main() -> None:
    """Entry point."""
    try:
        zip_path = create_zip()
        print(f"\n[OK] Submission archive ready: {zip_path}\n")
    except Exception as e:
        logger.critical(f"ZIP creation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
