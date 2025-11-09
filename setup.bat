@echo off
echo 🌸 Flower Classifier v2.0 - Setup Script
echo =========================================

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
cd backend
pip install --upgrade pip
pip install -r requirements.txt
cd ..

REM Create directories
echo Creating directories...
mkdir uploads
mkdir models

echo Setup complete!
echo.
echo To start:
echo 1. venv\Scripts\activate.bat
echo 2. python backend/app.py
echo 3. Open http://localhost:5000
pause
