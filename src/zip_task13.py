"""
zip_task13.py -- PlaceMux Phase 3, Task 13
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
    """Package Task 13 deliverables into a versioned ZIP archive."""
    task_num = 13
    date_str = datetime.date.today().strftime("%Y%m%d")
    zip_name = f"placemux_task{task_num:02d}_{date_str}.zip"
    logger.info(f"Creating ZIP: {zip_name}")
    included, skipped = [], []
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
        for folder in ["src", "models", "logs"]:
            if not os.path.isdir(folder):
                skipped.append(folder)
                continue
            for root, dirs, files in os.walk(folder):
                dirs[:] = [d for d in dirs if d != "__pycache__"]
                for file in files:
                    fp = os.path.join(root, file)
                    zf.write(fp)
                    included.append(fp)
        for f in [f"run_task{task_num:02d}.bat", "requirements.txt", "README.md"]:
            if os.path.exists(f):
                zf.write(f)
                included.append(f)
            else:
                skipped.append(f)
    logger.info(f"[OK] ZIP created: {zip_name}  ({len(included)} files)")
    if skipped:
        logger.warning(f"Skipped: {skipped}")
    return zip_name


def main() -> None:
    try:
        zip_path = create_zip()
        print(f"\n[OK] Submission archive ready: {zip_path}\n")
    except Exception as e:
        logger.critical(f"ZIP creation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
