@echo off
echo ========================================================
echo PlaceMux Phase 3: Task 17 - Partner API ^& ATS
echo ========================================================

if exist venv\Scripts\activate (
    call venv\Scripts\activate
) else (
    echo [WARNING] Virtual environment not found. Using system Python.
)

python src\demo_task17.py
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Task 17 failed. Check logs\task17.log
    pause
    exit /b 1
)

echo [SUCCESS] Task 17 completed successfully.
pause
