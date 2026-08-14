@echo off
echo =========================================================
echo PlaceMux Phase 3, Task 8: Retention, Cohorts ^& Churn
echo =========================================================

if exist venv\Scripts\activate (
    call venv\Scripts\activate
) else (
    echo [WARNING] No venv found. Running with global Python.
)

echo.
echo Running Churn Prediction End-to-End Demo...
echo.

python -m src.demo_task08

echo.
if %errorlevel% neq 0 (
    echo [ERROR] Demo failed. Check logs/task08_demo.log for details.
) else (
    echo [SUCCESS] Demo completed. Packaging ZIP...
    python -m src.zip_task08
)
echo.
pause
