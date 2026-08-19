@echo off
echo ========================================================
echo PlaceMux Phase 3 - Task 15 Launcher
echo Model Governance, Registry, and Drift Demo
echo ========================================================

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

echo.
echo [1/2] Running Initial Setup (train_task15.py)...
python src\train_task15.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Training setup failed. Check logs\task15.log.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [2/2] Running End-to-End Governance Demo (demo_task15.py)...
python src\demo_task15.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Demo failed. Check logs\task15.log.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ========================================================
echo TASK 15 PIPELINE COMPLETE
echo Please review logs\model_card_churn_model_v1.md
echo ========================================================
pause
