@echo off
setlocal
cd /d "%~dp0"
py -3.11 -m pip install -r requirements.txt
if errorlevel 1 goto :fail
py -3.11 stage2b_local_stac_prescreen.py --self-test
if errorlevel 1 goto :fail
py -3.11 stage2b_local_stac_prescreen.py --input inputs --connection-test
if errorlevel 1 goto :fail
echo.
echo CONNECTION TEST PASSED. You can run RUN_STAGE2B_LOCAL_STAC.bat
pause
exit /b 0
:fail
echo.
echo CONNECTION TEST FAILED. Read the error above.
pause
exit /b 1
