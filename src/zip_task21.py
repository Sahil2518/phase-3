"""
zip_task21.py — PlaceMux Task 21: Package deliverables into a ZIP archive.
"""
import os
import sys
import zipfile
import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR  = os.path.join(BASE_DIR, "src")

FILES_TO_ZIP = [
    os.path.join(SRC_DIR, "cost_model.py"),
    os.path.join(SRC_DIR, "cost_optimizer.py"),
    os.path.join(SRC_DIR, "demo_task21.py"),
    os.path.join(SRC_DIR, "zip_task21.py"),
    os.path.join(SRC_DIR, "semantic_search_engine.py"),
    os.path.join(SRC_DIR, "two_sided_recommender.py"),
    os.path.join(BASE_DIR, "economics_handoff.json"),
    os.path.join(BASE_DIR, "run_task21.bat"),
]


def create_zip() -> str:
    today = datetime.date.today().strftime("%Y%m%d")
    zip_name = f"placemux_task21_{today}.zip"
    zip_path = os.path.join(BASE_DIR, zip_name)

    created = []
    skipped = []

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in FILES_TO_ZIP:
            if os.path.exists(file_path):
                arcname = os.path.relpath(file_path, BASE_DIR)
                zf.write(file_path, arcname)
                created.append(arcname)
            else:
                skipped.append(file_path)

    print(f"\n[zip_task21] Archive created: {zip_path}")
    print(f"  Included ({len(created)}):")
    for f in created:
        print(f"    + {f}")
    if skipped:
        print(f"  Skipped ({len(skipped)}) — file not found:")
        for f in skipped:
            print(f"    - {f}")

    return zip_path


if __name__ == "__main__":
    create_zip()
