@echo off
setlocal
PowerShell -NoProfile -ExecutionPolicy Bypass -File "%~dp0RUN_FINAL_CONTEXT_PRODUCTION.ps1"
if errorlevel 1 (
  echo.
  echo FINAL CONTEXT PRODUCTION FAILED. Do not delete CONTEXT_SOURCE_CACHE. Read the exact error above.
  pause
  exit /b 1
)
echo.
echo FINAL CONTEXT PRODUCTION COMPLETE.
pause
