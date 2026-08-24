@echo off
echo ============================================================
echo  PlaceMux ^| Task 21 ^| Cost Optimization ^& FinOps
echo ============================================================

cd /d "%~dp0"

echo.
echo [Step 1/2] Running end-to-end demo...
call venv\Scripts\activate.bat
python src\demo_task21.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] demo_task21.py exited with errors. Check output above.
    pause
    exit /b 1
)

echo.
echo [Step 2/2] Packaging deliverables into ZIP...
python src\zip_task21.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] zip_task21.py failed.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Task 21 complete. ZIP archive ready for submission.
echo ============================================================
pause
