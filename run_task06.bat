@echo off
echo ===================================================
echo PlaceMux Phase 3, Task 6: Growth Instrumentation
echo ===================================================

if exist venv\Scripts\activate (
    call venv\Scripts\activate
) else (
    echo [WARNING] No venv found. Running with global Python.
)

echo.
echo Running Growth Simulation and Metric Demo...
echo.

python -m src.demo_growth

echo.
if %errorlevel% neq 0 (
    echo [ERROR] Demo failed. Check logs for details.
) else (
    echo [SUCCESS] Demo completed successfully.
)
echo.
pause
