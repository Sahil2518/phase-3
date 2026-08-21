import zipfile, os, datetime

task_num = 18
date_str = datetime.date.today().strftime("%Y%m%d")
zip_name = f"placemux_task{task_num:02d}_{date_str}.zip"

with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
    for folder in ["src", "models", "logs"]:
        if not os.path.exists(folder):
            continue
        for root, dirs, files in os.walk(folder):
            if "__pycache__" in root:
                continue
            for file in files:
                zf.write(os.path.join(root, file))
    for f in [f"run_task{task_num:02d}.bat", "requirements.txt", "README.md", f"zip_task{task_num:02d}.py"]:
        if os.path.exists(f):
            zf.write(f)

print(f"[SUCCESS] ZIP created: {zip_name}")
