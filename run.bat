@echo off
echo ====================================================
echo   Starting TCS AI FastAPI Backend Server...
echo ====================================================

cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found. Creating one...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    pip install -r requirements.txt
) else (
    call .venv\Scripts\activate.bat
)

python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
pause
