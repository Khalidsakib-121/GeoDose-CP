@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo GeoDose-CP Stage 3B v1.2 FREEZE controlled generator
echo ============================================================
PowerShell -NoProfile -ExecutionPolicy Bypass -File "%~dp0RUN_STAGE3B.ps1"
if errorlevel 1 goto :fail

echo.
echo SUCCESS. Upload GeoDose_Stage3B_OUTPUTS.zip for independent audit.
pause
exit /b 0

:fail
echo.
echo STAGE 3B FAILED. Read the error above.
pause
exit /b 1
