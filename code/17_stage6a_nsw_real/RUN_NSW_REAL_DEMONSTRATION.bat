@echo off
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0RUN_NSW_REAL_DEMONSTRATION.ps1"
set RC=%ERRORLEVEL%
echo.
if not "%RC%"=="0" (
  echo STAGE6A NSW REAL DEMONSTRATION FAILED. Keep _STAGE6A_WORK and the PowerShell log.
) else (
  echo STAGE6A RUN COMPLETE.
)
pause
exit /b %RC%
