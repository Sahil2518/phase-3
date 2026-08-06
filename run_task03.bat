@echo off
echo =======================================================
echo PlaceMux Phase 3 - Task 3: Profiling ^& Optimization
echo =======================================================

echo.
echo Running performance profiler and benchmark...
if exist venv\Scripts\activate (
    call venv\Scripts\activate
)
python src\demo_optimization.py

echo.
echo Packaging project into ZIP...
python src\zip_task03.py

echo.
echo =======================================================
echo Task 3 Complete! Output saved to logs/
echo =======================================================
pause
