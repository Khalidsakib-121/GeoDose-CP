@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Creating package-local Python 3.10 virtual environment...
  py -3.10 -m venv .venv || goto :fail
  .venv\Scripts\python.exe -m pip install --upgrade pip || goto :fail
  .venv\Scripts\python.exe -m pip install -r requirements_py310.txt || goto :fail
)
echo.
echo [1/5] Verifying frozen Python 3.10 runtime...
.venv\Scripts\python.exe verify_runtime.py || goto :fail
echo.
echo [2/5] Running unit tests...
.venv\Scripts\python.exe tests\test_integration.py || goto :fail
echo.
echo [3/5] Running exact M1-M6 integration gate...
.venv\Scripts\python.exe stage3d_d3_integration_run.py --overwrite || goto :fail
echo.
echo [4/5] Independently verifying outputs...
.venv\Scripts\python.exe verify_stage3d_d3_integration.py || goto :fail
echo.
echo [5/5] Packaging verified outputs...
.venv\Scripts\python.exe package_stage3d_d3_integration_outputs.py || goto :fail
echo.
echo SUCCESS. Send GeoDose_Stage3D_D3_M1_M6_EXACT_INTEGRATION_OUTPUTS.zip back to ChatGPT.
echo Do NOT start the 20-rep pilot yet.
pause
exit /b 0
:fail
echo.
echo FAILED. Stop here. Do not start pilot or modify frozen upstream stages.
pause
exit /b 1
