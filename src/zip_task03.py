import zipfile
import os
import datetime
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def main():
    try:
        task_num = 3
        date_str = datetime.date.today().strftime("%Y%m%d")
        zip_name = f"placemux_task{task_num:02d}_{date_str}.zip"
        
        with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
            for folder in ["src", "models", "logs"]:
                if not os.path.exists(folder):
                    continue
                for root, dirs, files in os.walk(folder):
                    for file in files:
                        # Exclude pycache
                        if "__pycache__" not in root:
                            zf.write(os.path.join(root, file))
                            
            for f in [f"run_task{task_num:02d}.bat", "requirements.txt", "README.md"]:
                if os.path.exists(f):
                    zf.write(f)
                    
        logger.info(f"✅ ZIP created successfully: {zip_name}")
    except Exception as e:
        logger.error(f"Failed to create ZIP: {e}")

if __name__ == "__main__":
    main()
