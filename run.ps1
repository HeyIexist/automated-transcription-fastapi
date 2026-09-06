Write-Host "====================================================" -ForegroundColor Cyan
Write-Host "   Starting TCS AI FastAPI Backend Server...       " -ForegroundColor Green
Write-Host "====================================================" -ForegroundColor Cyan

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$VenvPython = Join-Path $ScriptDir ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "[INFO] Creating Python virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
    & (Join-Path $ScriptDir ".venv\Scripts\pip.exe") install -r requirements.txt
}

Write-Host "[INFO] Launching FastAPI uvicorn server on http://localhost:8000..." -ForegroundColor Cyan
& $VenvPython -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
