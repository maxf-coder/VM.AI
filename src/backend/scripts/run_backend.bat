@echo off
REM VM.AI Backend Runner
REM Run this file to start the backend server

cd /d src\backend

echo Pre-downloading AI models...
uv run python scripts/download_models.py

echo Starting VM.AI Backend...
:: uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000

pause