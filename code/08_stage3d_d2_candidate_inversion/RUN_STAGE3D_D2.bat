@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0RUN_STAGE3D_D2.ps1"
set EXITCODE=%ERRORLEVEL%
echo.
if not "%EXITCODE%"=="0" (
  echo STAGE 3D D2 RUN FAILED WITH EXIT CODE %EXITCODE%.
) else (
  echo STAGE 3D D2 RUN FINISHED.
)
pause
exit /b %EXITCODE%
