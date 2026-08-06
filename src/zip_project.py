import zipfile
import os
import datetime
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

def create_zip():
    task_num = 1
    date_str = datetime.date.today().strftime("%Y%m%d")
    zip_name = f"placemux_task{task_num:02d}_{date_str}.zip"
    
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
        for folder in ["src", "models", "logs"]:
            if os.path.exists(folder):
                for root, dirs, files in os.walk(folder):
                    for file in files:
                        zf.write(os.path.join(root, file))
        for f in [f"run_task{task_num:02d}.bat", "requirements.txt", "README.md"]:
            if os.path.exists(f):
                zf.write(f)
                
    logging.info(f"✅ ZIP created: {zip_name}")

if __name__ == "__main__":
    create_zip()
