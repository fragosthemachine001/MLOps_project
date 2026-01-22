@echo off
echo Starting Credit Card Fraud Detection System...
echo.

REM Ensure we're in the project root directory
REM Check if we have the expected project structure
if not exist "src\credit_card_fraud_analysis" (
    echo Error: Please run this script from the MLOps_project root directory.
    echo Make sure you can see the 'src' and 'front' folders from where you're running this.
    pause
    exit /b 1
)

REM Use the current Python environment (base, conda env, venv, etc.)
echo Using current Python environment...
python --version
echo.

echo Starting Backend API Server on port 8000...
set PYTHONPATH=src
start "Backend API" cmd /k "python -m uvicorn src.credit_card_fraud_analysis.api:app --reload --host 0.0.0.0 --port 8000"

echo Waiting for backend to initialize...
timeout /t 5 /nobreak > nul

echo Starting Frontend on port 8501...
set BACKEND_URL=http://localhost:8000
start "Frontend" cmd /k "python -m streamlit run front/frontend.py --server.port=8501"

echo.
echo ========================================
echo   Credit Card Fraud Detection System
echo ========================================
echo Backend API: http://localhost:8000
echo Frontend:    http://localhost:8501
echo.
echo Both services are starting in separate windows.
echo The frontend will automatically open in your browser.
echo.
echo Press any key to exit this window...
pause > nul