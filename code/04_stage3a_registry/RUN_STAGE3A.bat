@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE=D:\GeoDose_Stage2B_Local_STAC_v1_0\.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"

echo ============================================================
echo GeoDose-CP Stage 3A v1.3.1 FREEZE: registry and generator audit
echo ============================================================
echo Python: %PYTHON_EXE%
echo This run uses no internet and does not run M1-M6.
echo.

"%PYTHON_EXE%" -c "import sys,importlib.metadata as m,numpy,pandas,geopandas,yaml; assert sys.version_info[:2]==(3,10), 'Python 3.10 is required'; expected={'numpy':'2.2.6','pandas':'2.3.3','geopandas':'1.1.4','PyYAML':'6.0.3','shapely':'2.1.2','pyogrio':'0.13.0'}; actual={k:m.version(k) for k in expected}; assert actual==expected, f'Frozen environment mismatch: {actual}'; print('Python',sys.version); print('Frozen environment check passed')"
if errorlevel 1 goto :missing

"%PYTHON_EXE%" tests\test_generator.py
if errorlevel 1 goto :fail

"%PYTHON_EXE%" stage3a_prepare.py --input inputs --output outputs_stage3a --overwrite
if errorlevel 1 goto :fail

"%PYTHON_EXE%" verify_stage3a.py --input inputs --output outputs_stage3a
if errorlevel 1 goto :fail

echo.
echo SUCCESS. Upload:
echo   %~dp0GeoDose_Stage3A_OUTPUTS.zip
echo.
pause
exit /b 0

:missing
echo.
echo Required packages are missing in this Python environment.
echo Run:
echo   "%PYTHON_EXE%" -m pip install -r requirements_frozen_py310.txt
echo Then rerun this batch file.
pause
exit /b 2

:fail
echo.
echo STAGE 3A FAILED. Read the error above.
pause
exit /b 1
