@echo off
echo ============================================================
echo  PlaceMux Phase 3 - Task 24
echo  Disaster Recovery, Chaos Testing and Business Continuity
echo ============================================================
echo.

cd /d "%~dp0"

if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    echo [WARN] No venv found - using system Python
)

echo [1/2] Running chaos demo...
python -m src.demo_task24
if %errorlevel% neq 0 (
    echo.
    echo [FAIL] Demo exited with errors. Check logs\task24.log for details.
    pause
    exit /b 1
)

echo.
echo [2/2] Packaging deliverables...
python src\zip_task24.py
if %errorlevel% neq 0 (
    echo [FAIL] ZIP packaging failed.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Task 24 Complete. Deliverables:
echo    logs\chaos_results.json
echo    logs\chaos_alerts.jsonl
echo    logs\ml_incident_runbook.md
echo    logs\task24.log
echo    placemux_task24_*.zip
echo ============================================================
pause
