@echo off
echo ========================================================
echo PlaceMux Phase 3: Task 16 - Multi-Tenancy ^& RBAC
echo ========================================================

if exist venv\Scripts\activate (
    call venv\Scripts\activate
) else (
    echo [WARNING] Virtual environment not found. Using system Python.
)

python src\demo_task16.py
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Task 16 failed. Check logs\task16.log
    pause
    exit /b 1
)

echo [SUCCESS] Task 16 completed successfully.
pause
