@echo off
echo =========================================================
echo PlaceMux Phase 3, Task 7: Activation ^& Onboarding Funnel
echo =========================================================

if exist venv\Scripts\activate (
    call venv\Scripts\activate
) else (
    echo [WARNING] No venv found. Running with global Python.
)

echo.
echo Running Cold-Start Recommendation Demo...
echo.

python -m src.demo_cold_start

echo.
if %errorlevel% neq 0 (
    echo [ERROR] Demo failed. Check logs for details.
) else (
    echo [SUCCESS] Demo completed successfully.
)
echo.
pause
