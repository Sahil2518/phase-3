@echo off
echo ============================================================
echo  PlaceMux Phase 3 -- Task 20: Enterprise Readiness Pilot
echo ============================================================

if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

echo [*] Running end-to-end enterprise pilot dry-run...
python src\demo_task20.py
if errorlevel 1 (
    echo.
    echo [ERROR] Task 20 demo failed. Check logs\task20.log for details.
    pause
    exit /b 1
)

echo.
echo [*] Packaging deliverables into ZIP...
python src\zip_task20.py
if errorlevel 1 (
    echo [WARN] ZIP packaging failed.
)

echo.
echo ============================================================
echo  Task 20 COMPLETE. Check logs\ for all output files.
echo ============================================================
pause
