@echo off
setlocal
set INPUT=D:\G3_STAGE1_MIN_AUDIT
set OUTPUT=D:\G3_STAGE2_BLOCKS

echo ============================================================
echo GeoDose-CP Stage 2A: 90 m blocks, queen graph and LOMO folds
echo ============================================================
echo Input : %INPUT%
echo Output: %OUTPUT%
echo.
echo The output folder will be recreated. Stage 1 files are read only.
echo Close QGIS, ArcGIS, Excel, or File Explorer previews using the output.
echo.

py -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 goto :fail

py "%~dp0stage2a_build_blocks.py" --self-test
if errorlevel 1 goto :fail

py "%~dp0stage2a_build_blocks.py" ^
  --input "%INPUT%" ^
  --output "%OUTPUT%" ^
  --overwrite
if errorlevel 1 goto :fail

py "%~dp0verify_stage2a.py" ^
  --input "%INPUT%" ^
  --output "%OUTPUT%"
if errorlevel 1 goto :fail

echo.
echo SUCCESS. Review:
echo   %OUTPUT%\stage2_block_summary.csv
echo   %OUTPUT%\STAGE2A_VERIFICATION.json
echo   %OUTPUT%\mine_blocks_90m.gpkg
echo.
pause
exit /b 0

:fail
echo.
echo STAGE 2A FAILED. Read the error above.
pause
exit /b 1
