@echo off
echo Starting Credit Card Fraud Detection API...
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

set PYTHONPATH=src
echo Starting API server on port 8000...
python -m uvicorn src.credit_card_fraud_analysis.api:app --reload --host 0.0.0.0 --port 8000
pause