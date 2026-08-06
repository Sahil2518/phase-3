@echo off
echo =======================================================
echo PlaceMux Phase 3 - Task 2: Observability ^& SLOs Demo
echo =======================================================

echo.
echo Running end-to-end SLO breach scenarios...
if exist venv\Scripts\activate (
    call venv\Scripts\activate
)
python src\demo_slo_breach.py

echo.
echo Packaging project into ZIP...
python src\zip_task02.py

echo.
echo =======================================================
echo Task 2 Complete! Output saved to logs/
echo =======================================================
pause
