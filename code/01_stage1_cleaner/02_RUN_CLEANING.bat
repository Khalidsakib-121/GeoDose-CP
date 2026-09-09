@echo off
setlocal
set ROOT=D:\G3_NSW_FRESH
set OUT=D:\G3_STAGE1_MIN_AUDIT

py "%~dp0stage1_clean_l03_l04.py" ^
  --root "%ROOT%" ^
  --output "%OUT%" ^
  --selection "%OUT%\selected_mines.csv" ^
  --overwrite

echo.
echo Completed. Review:
echo   %OUT%\cleaning_summary.csv
echo   %OUT%\all_record_qa.csv
echo   %OUT%\all_group_area_overlap_qa.csv
echo   %OUT%\L03_cleaned.gpkg
echo   %OUT%\L04_cleaned.gpkg
pause
