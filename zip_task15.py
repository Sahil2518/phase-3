"""
zip_task15.py — PlaceMux Phase 3, Task 15
=========================================
Packaging script for Task 15.
"""

import zipfile
import os
import datetime

def main():
    task_num = 15
    date_str = datetime.date.today().strftime("%Y%m%d")
    zip_name = f"placemux_task{task_num:02d}_{date_str}.zip"
    
    files_to_include = [
        "src/model_registry.py",
        "src/model_card.py",
        "src/train_task15.py",
        "src/demo_task15.py",
        "src/drift_monitor.py",
        "src/retraining_pipeline.py",
        "run_task15.bat",
        "zip_task15.py",
        "requirements.txt",
        "README.md"
    ]
    
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
        # Include specific files
        for f in files_to_include:
            if os.path.exists(f):
                zf.write(f)
                
        # Include directories
        for folder in ["models", "logs"]:
            if os.path.exists(folder):
                for root, dirs, files in os.walk(folder):
                    for file in files:
                        zf.write(os.path.join(root, file))
                        
    print(f"SUCCESS: ZIP created: {zip_name}")

if __name__ == "__main__":
    main()
