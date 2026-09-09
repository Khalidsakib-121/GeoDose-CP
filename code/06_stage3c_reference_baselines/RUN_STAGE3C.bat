@echo off
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0RUN_STAGE3C.ps1"
if errorlevel 1 (
  echo.
  echo STAGE 3C FAILED. Review the error above.
  pause
  exit /b 1
)
echo.
echo STAGE 3C RUN FINISHED.
pause
