@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo GeoDose-CP Stage 2B - Local DEA STAC / PyCharm-compatible run
echo ============================================================
echo.
echo This uses the public DEA STAC API and unsigned AWS S3 access.
echo No DEA Sandbox account is required.
echo.

py -3.11 -m pip install --upgrade pip
if errorlevel 1 goto :fail
py -3.11 -m pip install -r requirements.txt
if errorlevel 1 goto :fail

py -3.11 stage2b_local_stac_prescreen.py --self-test
if errorlevel 1 goto :fail

py -3.11 stage2b_local_stac_prescreen.py --input inputs --connection-test
if errorlevel 1 goto :fail

py -3.11 stage2b_local_stac_prescreen.py --input inputs --output outputs
if errorlevel 1 goto :fail

py -3.11 verify_stage2b_local_stac.py --input inputs --output outputs
if errorlevel 1 goto :fail

powershell -NoProfile -Command "Compress-Archive -Path outputs\* -DestinationPath GeoDose_Stage2B_OUTPUTS_LOCAL_STAC.zip -Force"
if errorlevel 1 goto :fail

echo.
echo STAGE 2B LOCAL STAC RUN FINISHED SUCCESSFULLY
echo Output folder: %~dp0outputs
echo Output ZIP   : %~dp0GeoDose_Stage2B_OUTPUTS_LOCAL_STAC.zip
echo.
pause
exit /b 0

:fail
echo.
echo STAGE 2B LOCAL STAC FAILED. Read the error above.
echo Existing verified checkpoints remain in the outputs folder.
pause
exit /b 1
