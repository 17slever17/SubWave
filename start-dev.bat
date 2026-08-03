@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-dev.ps1"
@REM $env:REALTIME_WEBUI_DEV_PORT=5180 .\start-dev.bat
