@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ================================================================
echo GeoDose-CP Stage5C Post-Freeze Computational Closure v1.0.0
echo ================================================================
echo This run does NOT train or retune models.
echo It verifies and freezes existing Stage5B + post-freeze evidence.
echo Completed closure stages are checkpointed and reused after interruption.
echo.

set "PYEXE="

where py >nul 2>nul
if %ERRORLEVEL%==0 (
    py -3.10 -c "import sys; assert sys.version_info[:2]==(3,10)" >nul 2>nul
    if %ERRORLEVEL%==0 set "PYEXE=py -3.10"
)

if not defined PYEXE (
    where python >nul 2>nul
    if %ERRORLEVEL%==0 (
        python -c "import sys; assert sys.version_info[:2]==(3,10)" >nul 2>nul
        if %ERRORLEVEL%==0 set "PYEXE=python"
    )
)

if not defined PYEXE (
    echo ERROR: Python 3.10 is required.
    echo Install Python 3.10 x64 and rerun this BAT file.
    pause
    exit /b 1
)

if not exist ".venv_stage5c\Scripts\python.exe" (
    echo Creating Stage5C Python environment...
    %PYEXE% -m venv .venv_stage5c
    if errorlevel 1 goto :fail
)

call ".venv_stage5c\Scripts\activate.bat"

echo Installing/verifying pinned closure dependencies...
python -m pip install --disable-pip-version-check --upgrade pip
if errorlevel 1 goto :fail
python -m pip install --disable-pip-version-check -r requirements_py310.txt
if errorlevel 1 goto :fail

echo.
echo Running Stage5C closure...
python run_stage5c_closure.py
if errorlevel 1 goto :fail

echo.
echo SUCCESS.
echo Final outputs are in:
echo   STAGE5C_CLOSURE_OUTPUTS
echo.
pause
exit /b 0

:fail
echo.
echo ================================================================
echo RUN INTERRUPTED OR FAILED
echo ================================================================
echo Do NOT delete _STAGE5C_WORK.
echo Rerun this BAT after fixing the reported issue.
echo Verified completed stages will be reused.
echo.
pause
exit /b 1
