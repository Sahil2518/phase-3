@echo off
echo =========================================================
echo PlaceMux Phase 3, Task 10: Growth Integration ^& Readout
echo =========================================================

if exist venv\Scripts\activate (
    call venv\Scripts\activate
) else (
    echo [WARNING] No venv found. Running with global Python.
)

echo.
echo Running Experiment Readout End-to-End Demo...
echo.

python -m src.demo_task10

echo.
if %errorlevel% neq 0 (
    echo [ERROR] Demo failed. Check logs/task10_demo.log for details.
) else (
    echo [SUCCESS] Demo completed. Packaging ZIP...
    python -m src.zip_task10
)
echo.
pause
