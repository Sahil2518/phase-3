@echo off
echo ========================================================
echo PlaceMux Phase 3: Task 18 - Enterprise Identity ^& SSO
echo ========================================================

if exist venv\Scripts\activate (
    call venv\Scripts\activate
) else (
    echo [WARNING] Virtual environment not found. Using system Python.
)

python src\demo_task18.py
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Task 18 failed. Check logs\task18.log
    pause
    exit /b 1
)

echo [SUCCESS] Task 18 completed successfully.
pause
