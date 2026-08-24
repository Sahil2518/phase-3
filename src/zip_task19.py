import os
import zipfile
import glob

def zip_project(output_filename="placemux_task19_submission.zip"):
    # Files specific to Task 19 and general requirements
    files_to_zip = [
        "src/tenant_manager.py",
        "src/admin_console.py",
        "src/multi_tenant_recommender.py",
        "src/demo_task19.py",
        "run_task19.bat"
    ]

    # Include any logs specifically generated
    if os.path.exists("logs/task19.log"):
        files_to_zip.append("logs/task19.log")
        
    print(f"Creating ZIP archive: {output_filename}")
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in files_to_zip:
            if os.path.exists(file):
                print(f"  Adding: {file}")
                zipf.write(file)
            else:
                print(f"  [WARNING] File not found: {file}")

    print(f"\nSuccessfully created {output_filename}")

if __name__ == "__main__":
    zip_project()
