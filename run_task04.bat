@echo off
echo ===================================================
echo PlaceMux - Phase 3, Task 4
echo Load Testing ^& Scaling Plan Demo
echo ===================================================

echo.
echo Activating Virtual Environment...
call venv\Scripts\activate.bat

echo.
echo Running Load Test Orchestrator...
python src\demo_load_test.py

echo.
echo Packaging Deliverables...
python src\zip_task04.py

echo.
echo ===================================================
echo Task 4 Execution Complete.
echo ===================================================
pause
