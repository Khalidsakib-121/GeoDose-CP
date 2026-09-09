@echo off
setlocal
set ROOT=D:\G3_NSW_FRESH
set OUT=D:\G3_STAGE1_MIN_AUDIT

py -m pip install -r "%~dp0requirements.txt"
py "%~dp0stage1_clean_l03_l04.py" ^
  --root "%ROOT%" ^
  --output "%OUT%" ^
  --prepare-selection

echo.
echo Open %OUT%\selected_mines.csv. Prioritize has_consecutive_common_pair=TRUE, then set include=1 for exactly five mines.
pause
