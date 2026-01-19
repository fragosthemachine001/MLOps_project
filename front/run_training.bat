@echo off
echo Starting Credit Card Fraud Detection Training...
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

echo Starting training...
set PYTHONPATH=src
python src/credit_card_fraud_analysis/train_lightning.py
pause