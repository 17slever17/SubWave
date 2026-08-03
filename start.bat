@echo off
setlocal
cd /d "%~dp0"
title Realtime Subtitle Translator

echo [webui] Realtime Subtitle Translator
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\bootstrap.ps1"
if errorlevel 1 (
  echo.
  echo [webui] Setup failed. Review the error above.
  pause
  exit /b 1
)

".venv\Scripts\python.exe" "scripts\launch_webui.py"
if errorlevel 1 (
  echo.
  echo [webui] The application stopped with an error.
  pause
)
