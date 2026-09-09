@echo off
setlocal EnableExtensions
set "ROOT=%~dp0"
set "DEST=D:\GeoDose_Stage3E_N2_N3_Scalable_Certification_v1_0"
set "NEEDS_RELOCATE=0"
echo(%ROOT%| findstr /I /C:"\AppData\Local\Temp\" >nul && set "NEEDS_RELOCATE=1"
echo(%ROOT%| findstr /I /C:".zip." >nul && set "NEEDS_RELOCATE=1"
if "%NEEDS_RELOCATE%"=="1" goto :relocate
goto :run_local

:relocate
echo.
echo Detected execution from a ZIP/compressed-folder temporary location.
echo Relocating the complete Stage3E package to D:\ before Python starts.
if not exist "D:\" (
  echo ERROR: D:\ is unavailable. Use Extract All to a normal folder and run again.
  goto :fail
)
if not exist "%ROOT%preflight_windows.py" goto :incomplete
if not exist "%ROOT%stage3e_run.py" goto :incomplete
if not exist "%ROOT%configs\stage3e_contract.yaml" goto :incomplete
if not exist "%ROOT%src\geodose_stage3e\runner.py" goto :incomplete
if not exist "%ROOT%tests\test_stage3e.py" goto :incomplete
if not exist "%DEST%" mkdir "%DEST%" || goto :fail
robocopy "%ROOT%" "%DEST%" /E /R:2 /W:1 /NFL /NDL /NJH /NJS /NP /XD ".venv" "outputs_stage3e" /XF "GeoDose_Stage3E_N2_N3_SCALABLE_CERTIFICATION_OUTPUTS.zip" >nul
if errorlevel 8 goto :fail
if not exist "%DEST%\stage3e_run.py" goto :fail
call "%DEST%\RUN_STAGE3E.bat"
exit /b %errorlevel%

:incomplete
echo ERROR: Windows did not materialize the full ZIP contents.
echo Right-click the ZIP, choose Extract All, extract to D:\, and run RUN_STAGE3E.bat from the extracted project folder.
goto :fail

:run_local
cd /d "%ROOT%"
echo.
echo [PRECHECK] Verifying accepted inputs, exact hashes, ZIP CRCs, D2 source 41/41, and accepted D4 status...
py -3.10 preflight_windows.py || goto :fail

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
echo [2/5] Running Stage3E mathematical and implementation unit tests...
.venv\Scripts\python.exe tests\test_stage3e.py || goto :fail

echo.
echo [3/5] Running N2/N3 scalable certification gate...
.venv\Scripts\python.exe stage3e_run.py --overwrite || goto :fail

echo.
echo [4/5] Independently verifying Stage3E outputs...
.venv\Scripts\python.exe verify_stage3e.py || goto :fail

echo.
echo [5/5] Packaging verified Stage3E outputs...
.venv\Scripts\python.exe package_stage3e_outputs.py || goto :fail

echo.
echo SUCCESS. Send GeoDose_Stage3E_N2_N3_SCALABLE_CERTIFICATION_OUTPUTS.zip back to ChatGPT.
echo Do NOT start the 20-rep pilot or production run until the Windows Stage3E output is independently reviewed.
pause
exit /b 0

:fail
echo.
echo FAILED. Stop here. Do not modify frozen upstream stages and do not start the pilot.
pause
exit /b 1
