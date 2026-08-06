@echo off
echo ========================================================
echo PlaceMux Phase 3 - Task 1 Pipeline
echo ========================================================

REM Activate virtual environment if present
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

echo.
echo [1/5] Simulating Live Traffic Logs...
python src\data_simulator.py
if %ERRORLEVEL% neq 0 goto :error

echo.
echo [2/5] Generating Model Health Report...
python src\health_report.py
if %ERRORLEVEL% neq 0 goto :error

echo.
echo [3/5] Identifying Intelligence Defects...
python src\defect_finder.py
if %ERRORLEVEL% neq 0 goto :error

echo.
echo [4/5] Generating Phase-3 Backlog...
python src\backlog_creator.py
if %ERRORLEVEL% neq 0 goto :error

echo.
echo [5/5] Running Worked Example Demo...
python src\demo_example.py
if %ERRORLEVEL% neq 0 goto :error

echo.
echo Packaging Deliverables into ZIP...
python src\zip_project.py

echo.
echo ========================================================
echo SUCCESS! Pipeline completed.
echo ========================================================
pause
exit /b 0

:error
echo.
echo ========================================================
echo FAILURE! Pipeline aborted due to an error.
echo ========================================================
pause
exit /b 1
