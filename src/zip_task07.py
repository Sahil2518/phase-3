import zipfile
import os
import datetime
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def create_zip():
    task_num = 7
    date_str = datetime.date.today().strftime("%Y%m%d")
    zip_name = f"placemux_task{task_num:02d}_{date_str}.zip"
    
    files_to_include = []
    
    # Add src files
    for root, dirs, files in os.walk("src"):
        for file in files:
            if file.endswith(".py"):
                files_to_include.append(os.path.join(root, file))
                
    # Add log files
    if os.path.exists("logs"):
        for root, dirs, files in os.walk("logs"):
            for file in files:
                files_to_include.append(os.path.join(root, file))
                
    # Add top level files
    for f in ["run_task07.bat", "README.md", "requirements.txt"]:
        if os.path.exists(f):
            files_to_include.append(f)
            
    try:
        with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in files_to_include:
                zf.write(file_path)
        logger.info(f"✅ ZIP created successfully: {zip_name}")
    except Exception as e:
        logger.error(f"Failed to create ZIP: {e}")
        sys.exit(1)

if __name__ == "__main__":
    create_zip()
