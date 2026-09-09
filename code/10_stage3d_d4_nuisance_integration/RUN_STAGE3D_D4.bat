@echo off
setlocal EnableExtensions
set "ROOT=%~dp0"

rem ---------------------------------------------------------------------------
rem Robust Windows compressed-folder guard.
rem If the BAT is launched directly from a ZIP in Explorer, Windows materializes
rem it under AppData\Local\Temp (often in a path containing .zip.<suffix>).
rem Creating a venv there is unreliable. Relocate the complete package to D:\
rem and relaunch before touching Python or frozen scientific inputs.
rem ---------------------------------------------------------------------------
set "NEEDS_RELOCATE=0"
echo(%ROOT%| findstr /I /C:"\AppData\Local\Temp\" >nul && set "NEEDS_RELOCATE=1"
echo(%ROOT%| findstr /I /C:".zip." >nul && set "NEEDS_RELOCATE=1"

if "%NEEDS_RELOCATE%"=="1" goto :relocate

goto :run_local

:relocate
echo.
echo Detected execution from a ZIP/compressed-folder temporary location.
echo This launcher will safely copy the package to D: and run it there.
echo.
if not exist "D:\" (
  echo ERROR: D:\ is not available. Extract the ZIP manually to D:\ and run again.
  goto :fail
)

rem Require the files needed for a complete scientific run before copying.
if not exist "%ROOT%preflight_windows.py" goto :incomplete_temp
if not exist "%ROOT%stage3d_d4_run.py" goto :incomplete_temp
if not exist "%ROOT%requirements_py310.txt" goto :incomplete_temp
if not exist "%ROOT%verify_runtime.py" goto :incomplete_temp
if not exist "%ROOT%verify_stage3d_d4.py" goto :incomplete_temp
if not exist "%ROOT%configs\d4_contract.yaml" goto :incomplete_temp
if not exist "%ROOT%src\geodose_stage3d_d4\runner.py" goto :incomplete_temp
if not exist "%ROOT%tests\test_d4.py" goto :incomplete_temp

set "DEST=D:\GeoDose_Stage3D_D4_M6_NewData_Nuisance_Integration_v1_1_0"
echo Relocating package to:
echo   %DEST%
if not exist "%DEST%" mkdir "%DEST%" || goto :fail

rem ROBOCOPY success codes are 0-7; 8+ means failure.
robocopy "%ROOT%" "%DEST%" /E /R:2 /W:1 /NFL /NDL /NJH /NJS /NP /XD ".venv" "outputs_stage3d_d4" /XF "GeoDose_Stage3D_D4_M6_NEWDATA_NUISANCE_OUTPUTS.zip" >nul
if errorlevel 8 (
  echo ERROR: Could not copy the complete package out of the ZIP/temp location.
  goto :fail
)

if not exist "%DEST%\stage3d_d4_run.py" goto :relocate_verify_fail
if not exist "%DEST%\src\geodose_stage3d_d4\runner.py" goto :relocate_verify_fail

echo Relocation complete. Relaunching from D:\ ...
echo.
call "%DEST%\RUN_STAGE3D_D4.bat"
exit /b %errorlevel%

:incomplete_temp
echo ERROR: Windows did not materialize the full package from the ZIP.
echo Please right-click the ZIP, choose Extract All, extract to D:\, and run RUN_STAGE3D_D4.bat from the extracted folder.
goto :fail

:relocate_verify_fail
echo ERROR: Relocation verification failed. The copied package is incomplete.
goto :fail

:run_local
cd /d "%ROOT%"

echo.
echo [PRECHECK] Verifying frozen Windows paths, hashes, ZIP integrity, and source trees before environment setup...
py -3.10 preflight_windows.py || goto :fail

rem If a previous interrupted run left a partial venv, rebuild it rather than
rem trusting only the existence of python.exe.
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" verify_runtime.py >nul 2>&1
  if errorlevel 1 (
    echo Existing package-local virtual environment is incomplete or incompatible. Rebuilding...
    rmdir /s /q ".venv"
  )
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
echo [2/5] Running D4 unit tests...
.venv\Scripts\python.exe tests\test_d4.py || goto :fail
echo.
echo [3/5] Running source-level M6 new-data + nuisance integration gate...
.venv\Scripts\python.exe stage3d_d4_run.py --overwrite || goto :fail
echo.
echo [4/5] Independently verifying D4 outputs...
.venv\Scripts\python.exe verify_stage3d_d4.py || goto :fail
echo.
echo [5/5] Packaging verified outputs...
.venv\Scripts\python.exe package_stage3d_d4_outputs.py || goto :fail
echo.
echo SUCCESS. Send GeoDose_Stage3D_D4_M6_NEWDATA_NUISANCE_OUTPUTS.zip back to ChatGPT.
echo Do NOT start Stage3E, the 20-rep pilot, or production runs until this Windows output is independently reviewed.
pause
exit /b 0

:fail
echo.
echo FAILED. Stop here. Do not change frozen upstream stages and do not start the pilot.
pause
exit /b 1
