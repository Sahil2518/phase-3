@echo off
echo =========================================================
echo PlaceMux Phase 3, Task 19: White-Label Configurability
echo =========================================================

if exist venv\Scripts\activate (
    call venv\Scripts\activate
) else (
    echo [WARNING] No venv found. Running with global Python.
)

echo.
echo Running Task 19 End-to-End Demo...
echo.

python src\demo_task19.py

echo.
if %errorlevel% neq 0 (
    echo [ERROR] Demo failed. Check logs/task19.log for details.
) else (
    echo [SUCCESS] Demo completed. Packaging ZIP...
    python src\zip_task19.py
)
echo.
pause
