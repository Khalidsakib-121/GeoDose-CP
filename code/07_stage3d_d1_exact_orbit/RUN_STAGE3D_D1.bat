@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0RUN_STAGE3D_D1.ps1"
if errorlevel 1 (
  echo.
  echo STAGE 3D D0/D1 FAILED. Review the error above.
  pause
  exit /b 1
)
echo.
echo STAGE 3D D0/D1 RUN FINISHED.
pause
