@echo off
echo =========================================================
echo PlaceMux Phase 3, Task 23: Compliance Audit
echo DPDP, GDPR ^& SOC 2 Readiness
echo =========================================================

if exist venv\Scripts\activate (
    call venv\Scripts\activate
) else (
    echo [WARNING] No venv found. Running with global Python.
)

echo.
echo Running Task 23 Compliance Demo (Stages A-E)...
echo.

python -m src.demo_task23

echo.
if %errorlevel% neq 0 (
    echo [ERROR] Compliance demo failed. Check logs/task23.log
) else (
    echo [SUCCESS] Demo completed. Packaging ZIP...
    python -m src.zip_task23
)
echo.
pause
