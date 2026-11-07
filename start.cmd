@echo off
setlocal
chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 (
  echo [kconv] uv not found.
  echo [kconv] Install it from https://docs.astral.sh/uv/ and run this file again.
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [kconv] First launch: installing dependencies...
  uv sync
  if errorlevel 1 (
    echo [kconv] uv sync failed.
    exit /b 1
  )
)

if not exist ".venv\Scripts\kaggle.exe" (
  echo [kconv] Kaggle CLI is missing from .venv; installing dependencies...
  uv sync --inexact
  if errorlevel 1 (
    echo [kconv] uv sync failed; Kaggle submit will be unavailable.
  )
)

if not exist "data" mkdir "data"

echo [kconv] Starting kconv studio...
echo [kconv] URL:    http://127.0.0.1:8765/ (or --host/--port)
echo [kconv] Data:   %CD%\data
echo [kconv] Stop:   Ctrl+C
echo.

uv run --no-sync kconv web --data data %*
set "KCONV_EXIT=%ERRORLEVEL%"
if not "%KCONV_EXIT%"=="0" (
  echo.
  echo [kconv] kconv web exited with code %KCONV_EXIT%.
  echo [kconv] If the port is busy, try: start.cmd --port 8766
)
exit /b %KCONV_EXIT%
