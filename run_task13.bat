@echo off
echo =========================================================
echo PlaceMux Phase 3, Task 13: Semantic Search and Vector Retrieval
echo =========================================================

if exist venv\Scripts\activate (
    call venv\Scripts\activate
) else (
    echo [WARNING] No venv found. Running with global Python.
)

echo.
echo Running Task 13 End-to-End Demo...
echo.

python -m src.demo_task13

echo.
if %errorlevel% neq 0 (
    echo [ERROR] Demo failed.
) else (
    echo [SUCCESS] Demo completed. Packaging ZIP...
    python -m src.zip_task13
)
echo.
pause
