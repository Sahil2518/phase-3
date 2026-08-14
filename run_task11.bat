@echo off
echo =========================================================
echo PlaceMux Phase 3, Task 11: Learning-to-Rank (LTR)
echo =========================================================

if exist venv\Scripts\activate (
    call venv\Scripts\activate
) else (
    echo [WARNING] No venv found. Running with global Python.
)

echo.
echo Running LTR End-to-End Demo...
echo.

python -m src.demo_task11

echo.
if %errorlevel% neq 0 (
    echo [ERROR] Demo failed. Check logs/task11_demo.log for details.
) else (
    echo [SUCCESS] Demo completed. Packaging ZIP...
    python -m src.zip_task11
)
echo.
pause
