@echo off
setlocal EnableExtensions
set "ROOT=%~dp0"
set "DEST=D:\GeoDose_Stage4_Production_F06_Ablation_Efficiency_v1_0"
set "NEEDS_RELOCATE=0"
echo(%ROOT%| findstr /I /C:"\AppData\Local\Temp\" >nul && set "NEEDS_RELOCATE=1"
echo(%ROOT%| findstr /I /C:".zip." >nul && set "NEEDS_RELOCATE=1"
if "%NEEDS_RELOCATE%"=="1" goto :relocate
goto :run_local
:relocate
echo.
echo Detected execution from a ZIP/compressed-folder temporary location.
echo Relocating the complete Stage4 package to D:\ before Python starts.
if not exist "D:\" (echo ERROR: D:\ is unavailable. Use Extract All to a normal folder and run again.& goto :fail)
if not exist "%ROOT%preflight_windows.py" goto :incomplete
if not exist "%ROOT%stage4_run.py" goto :incomplete
if not exist "%ROOT%configs\production_contract.yaml" goto :incomplete
if not exist "%ROOT%src\geodose_stage4\runner.py" goto :incomplete
if not exist "%ROOT%tests\test_stage4.py" goto :incomplete
if not exist "%DEST%" mkdir "%DEST%" || goto :fail
robocopy "%ROOT%" "%DEST%" /E /R:2 /W:1 /NFL /NDL /NJH /NJS /NP /XD ".venv" "outputs_stage4" /XF "GeoDose_Stage4_PRODUCTION_F06_ABLATION_EFFICIENCY_OUTPUTS.zip" >nul
if errorlevel 8 goto :fail
if not exist "%DEST%\stage4_run.py" goto :fail
call "%DEST%\RUN_STAGE4.bat"
exit /b %errorlevel%
:incomplete
echo ERROR: Windows did not materialize the full ZIP contents.
echo Right-click the ZIP, choose Extract All, extract to D:\, then run RUN_STAGE4.bat.
goto :fail
:run_local
cd /d "%ROOT%"
echo.
echo [PRECHECK] Verifying Stage3F freeze and every accepted upstream archive/source tree BEFORE environment setup...
py -3.10 preflight_windows.py || goto :fail
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" verify_runtime.py >nul 2>&1
  if errorlevel 1 (echo Existing package-local virtual environment is incompatible. Rebuilding...& rmdir /s /q ".venv")
)
if not exist ".venv\Scripts\python.exe" (
  echo Creating package-local Python 3.10 virtual environment...
  py -3.10 -m venv .venv || goto :fail
  .venv\Scripts\python.exe -m pip install --upgrade pip || goto :fail
  .venv\Scripts\python.exe -m pip install -r requirements_py310.txt || goto :fail
)
echo.
echo [1/5] Verifying frozen Python 3.10 runtime...
.venv\Scripts\python.exe verify_runtime.py || goto :fail
echo.
echo [2/5] Running Stage4 mathematical, API, F06, registry, provenance, and refusal tests...
.venv\Scripts\python.exe tests\test_stage4.py || goto :fail
echo.
echo [3/5] Running full controlled production + F06 + ablation/efficiency...
echo Main production: 3000 case-replications. F06: 600. Registered stresses: 590.
echo This is a long run. One atomic checkpoint is saved after every case-replication.
echo If Windows/session stops, run this BAT again: completed checkpoints resume automatically.
.venv\Scripts\python.exe stage4_run.py || goto :fail
echo.
echo [4/5] Independently verifying complete Stage4 outputs...
.venv\Scripts\python.exe verify_stage4.py || goto :fail
echo.
echo [5/5] Packaging verified Stage4 outputs...
.venv\Scripts\python.exe package_stage4_outputs.py || goto :fail
echo.
echo SUCCESS. Send GeoDose_Stage4_PRODUCTION_F06_ABLATION_EFFICIENCY_OUTPUTS.zip back to ChatGPT.
echo Do NOT start MineDoseBench until this Windows production output is independently reviewed.
pause
exit /b 0
:fail
echo.
echo FAILED. Do not change Stage3F thresholds or frozen upstream stages.
echo If the failure occurred during [3/5], rerun this BAT after correcting only Stage4; completed checkpoints will resume.
pause
exit /b 1
