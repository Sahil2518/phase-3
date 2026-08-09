@echo off
echo ===================================================
echo PlaceMux - Phase 3, Task 5
echo Capstone Reliability Sign-off
echo ===================================================

echo.
echo Activating Virtual Environment...
call venv\Scripts\activate.bat

echo.
echo Running Capstone Orchestrator...
python src\demo_signoff.py

echo.
echo Packaging Deliverables...
python src\zip_task05.py

echo.
echo ===================================================
echo Task 5 Execution Complete.
echo ===================================================
pause
