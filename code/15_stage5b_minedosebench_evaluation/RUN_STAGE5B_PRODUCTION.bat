@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0RUN_STAGE5B_PRODUCTION.ps1"
set EC=%ERRORLEVEL%
if not "%EC%"=="0" (
  echo.
  echo STAGE5B PRODUCTION/VERIFICATION FAILED. Check the PowerShell output above.
) else (
  echo.
  echo STAGE5B RUN COMPLETE.
)
pause
exit /b %EC%
